
"""Optuna architecture tuning for the full-context Hybrid LSTM."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.features import PREGAME_FEATURES
from src.modeling.lstm_hybrid_experiment import (
    prepare_context,
    prepare_sequences,
)
from src.modeling.lstm_hybrid_model import DraftKingsHybridLSTM
from src.modeling.model_adapters.lstm_adapter import load_historical_sequences
from src.modeling.splits import biweekly_walk_forward_splits


FOLDS = (2, 4, 6)
SEQUENCE_LENGTH = 10
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    df, raw, lengths = load_historical_sequences()
    raw = raw[:, -SEQUENCE_LENGTH:, :].copy()
    lengths = np.minimum(lengths, SEQUENCE_LENGTH).astype(np.int64)

    dates = pd.to_datetime(df["GAME_DATE"])
    y = pd.to_numeric(df["DK_POINTS"]).to_numpy(dtype=np.float32)

    prepared = []

    for fold, (train_labels, val_labels) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        if fold not in FOLDS:
            continue

        train_idx = df.index.get_indexer(train_labels)
        val_idx = df.index.get_indexer(val_labels)

        unique_dates = np.sort(dates.iloc[train_idx].unique())
        cutoff = unique_dates[max(1, int(len(unique_dates) * 0.9)) - 1]

        fit_idx = train_idx[dates.iloc[train_idx].to_numpy() <= cutoff]
        stop_idx = train_idx[dates.iloc[train_idx].to_numpy() > cutoff]

        X = prepare_sequences(raw, lengths, fit_idx)
        C = prepare_context(df, fit_idx, PREGAME_FEATURES)

        y_mean = float(y[fit_idx].mean())
        y_std = max(float(y[fit_idx].std()), 1.0)

        def dataset(indices):
            return TensorDataset(
                torch.from_numpy(X[indices]),
                torch.from_numpy(lengths[indices]),
                torch.from_numpy(C[indices]),
                torch.from_numpy((y[indices] - y_mean) / y_std),
            )

        prepared.append({
            "fold": fold,
            "fit": dataset(fit_idx),
            "stop": dataset(stop_idx),
            "val": dataset(val_idx),
            "actual": y[val_idx],
            "y_mean": y_mean,
            "y_std": y_std,
            "sequence_features": X.shape[2],
            "context_features": C.shape[1],
        })

        print(
            f"Prepared fold {fold}: fit={len(fit_idx):,}, "
            f"stop={len(stop_idx):,}, val={len(val_idx):,}",
            flush=True,
        )

    return prepared


def evaluate_fold(data, params, device):
    fold = data["fold"]
    set_seed(42 + fold)

    fit_loader = DataLoader(
        data["fit"], batch_size=BATCH_SIZE, shuffle=True
    )
    stop_loader = DataLoader(
        data["stop"], batch_size=BATCH_SIZE
    )
    val_loader = DataLoader(
        data["val"], batch_size=BATCH_SIZE
    )

    model = DraftKingsHybridLSTM(
        sequence_features=data["sequence_features"],
        context_features=data["context_features"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for xb, lb, cb, yb in fit_loader:
            optimizer.zero_grad()
            prediction = model(
                xb.to(device), lb.to(device), cb.to(device)
            )
            loss = loss_fn(prediction, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        stop_loss = 0.0

        with torch.no_grad():
            for xb, lb, cb, yb in stop_loader:
                prediction = model(
                    xb.to(device), lb.to(device), cb.to(device)
                )
                stop_loss += (
                    loss_fn(prediction, yb.to(device)).item() * len(yb)
                )

        stop_loss /= len(data["stop"])

        if stop_loss < best_loss - 0.0001:
            best_loss = stop_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    predictions = []

    with torch.no_grad():
        for xb, lb, cb, _ in val_loader:
            predictions.append(
                model(
                    xb.to(device), lb.to(device), cb.to(device)
                ).cpu().numpy()
            )

    predicted = np.maximum(
        np.concatenate(predictions) * data["y_std"] + data["y_mean"],
        0.0,
    )

    absolute_error = np.abs(predicted - data["actual"])

    return {
        "fold": fold,
        "mae": float(absolute_error.mean()),
        "absolute_error_sum": float(absolute_error.sum()),
        "n": len(absolute_error),
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    output = Path("reports/modeling/lstm/optuna")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)

    prepared = prepare_data()

    study = optuna.create_study(
        study_name="hybrid_lstm_full_seq10_v1",
        storage=f"sqlite:///{(output / 'study.db').resolve().as_posix()}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )

    if not study.trials:
        study.enqueue_trial({
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.20,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
        })

    def objective(trial):
        params = {
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            ),
            "num_layers": trial.suggest_categorical(
                "num_layers", [1, 2]
            ),
            "dropout": trial.suggest_float(
                "dropout", 0.10, 0.35
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0003, 0.003, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 0.00001, 0.01, log=True
            ),
        }

        total_error = 0.0
        total_n = 0

        for step, data in enumerate(prepared):
            result = evaluate_fold(data, params, device)

            total_error += result["absolute_error_sum"]
            total_n += result["n"]

            current_mae = total_error / total_n

            trial.report(current_mae, step)

            print(
                f"Trial {trial.number} | fold {result['fold']} | "
                f"MAE={result['mae']:.4f} | "
                f"running={current_mae:.4f} | "
                f"epoch={result['best_epoch']}",
                flush=True,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        return total_error / total_n

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
    )

    print("\nBEST TRIAL")
    print(f"MAE: {study.best_value:.4f}")
    print(f"Parameters: {study.best_params}")

    study.trials_dataframe().to_csv(
        output / "trials.csv", index=False
    )


if __name__ == "__main__":
    main()
