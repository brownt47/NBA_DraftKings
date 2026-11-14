"""Seven-fold walk-forward experiments for the DraftKings hybrid LSTM."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.features import PREGAME_FEATURES, CATEGORICAL_FEATURES
from src.modeling.lstm_hybrid_model import DraftKingsHybridLSTM
from src.modeling.model_adapters.lstm_adapter import load_historical_sequences
from src.modeling.splits import biweekly_walk_forward_splits

SMALL_CONTEXT = [
    "OPPONENT", "HOME_AWAY", "DAYS_REST", "IS_BACK_TO_BACK",
    "OPP_DAYS_REST", "OPP_IS_BACK_TO_BACK", "MATCHUP_PACE_LAST_5",
    "OPP_DEF_RATING_LAST_5", "TRAVEL_MILES", "PLAYER_MINUTES_LAST_7_DAYS",
]


def prepare_context(df, fit_idx, columns):
    """Fit preprocessing on the fitting portion only, then transform all rows."""
    parts = []
    for column in columns:
        if column in CATEGORICAL_FEATURES:
            train_values = df.iloc[fit_idx][column].fillna("UNKNOWN").astype(str)
            categories = sorted(train_values.unique())
            values = df[column].fillna("UNKNOWN").astype(str)
            encoded = np.column_stack([
                (values == category).to_numpy(dtype=np.float32)
                for category in categories
            ])
            parts.append(encoded)
        else:
            values = pd.to_numeric(df[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            training = values.iloc[fit_idx]
            median = training.median()
            if pd.isna(median):
                median = 0.0
            training_filled = training.fillna(median)
            mean = float(training_filled.mean())
            std = float(training_filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            scaled = ((values.fillna(median) - mean) / std).to_numpy(dtype=np.float32)
            parts.append(scaled[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def prepare_sequences(raw, lengths, fit_idx):
    """Fit sequence scaling on real historical timesteps, never padding."""
    steps = raw.shape[1]
    mask = np.arange(steps)[None, :] >= steps - lengths[fit_idx, None]
    observations = raw[fit_idx][mask]
    if not len(observations):
        raise ValueError("No historical observations available for sequence scaling")
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = np.nan_to_num((raw - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    mask_all = np.arange(steps)[None, :] >= steps - lengths[:, None]
    scaled[~mask_all] = 0.0
    return scaled.astype(np.float32)


def score(actual, predicted):
    errors = predicted - actual
    return {
        "N": len(actual),
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "BIAS": float(errors.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=["small", "full"], default="small")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        choices=range(5, 11),
        help="Maximum number of historical games used by the LSTM",
    )
    parser.add_argument(
        "--holdout-april",
        action="store_true",
        help="Evaluate April 2026 using only pre-April training data",
    )
    args = parser.parse_args()
    columns = SMALL_CONTEXT if args.context == "small" else PREGAME_FEATURES
    output = args.output_dir or (
        Path("reports/modeling/lstm")
        / "sequence_tuning"
        / f"hybrid_{args.context}_seq_{args.sequence_length}"
    )
    output.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Context: {args.context} ({len(columns)} original columns)", flush=True)

    df, raw_sequences, lengths = load_historical_sequences()

    # Keep only the most recent N historical games.
    # Sequences are right-aligned, with padding on the left.
    raw_sequences = raw_sequences[:, -args.sequence_length:, :].copy()

    # Update lengths so packed sequences do not exceed the new window.
    lengths = np.minimum(lengths, args.sequence_length).astype(np.int64)

    assert raw_sequences.shape[1] == args.sequence_length
    assert np.all(lengths <= args.sequence_length)
    assert np.all(lengths >= 0)

    print(
        f"Sequence length: {args.sequence_length} | "
        f"Sequence shape: {raw_sequences.shape}",
        flush=True,
    )

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing context columns: {missing}")
    dates = pd.to_datetime(df["GAME_DATE"])
    y = pd.to_numeric(df["DK_POINTS"], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite DK_POINTS targets")

    all_predictions = []
    fold_metrics = []

    if args.holdout_april:
        split_iterator = biweekly_walk_forward_splits(
            df,
            start_date="2026-04-01",
            end_date="2026-05-01",
            window_days=30,
        )
    else:
        split_iterator = biweekly_walk_forward_splits(df)

    for fold, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        if fold not in args.folds:
            continue
        # Reset random seeds for reproducible comparisons across sequence lengths.
        fold_seed = 42 + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        # The current dataset uses a RangeIndex, so label indices equal row positions.
        train_idx = df.index.get_indexer(train_idx)
        val_idx = df.index.get_indexer(val_idx)
        unique_dates = np.sort(dates.iloc[train_idx].unique())
        cutoff = unique_dates[max(1, int(len(unique_dates) * 0.9)) - 1]
        fit_idx = train_idx[dates.iloc[train_idx].to_numpy() <= cutoff]
        stop_idx = train_idx[dates.iloc[train_idx].to_numpy() > cutoff]
        if not len(fit_idx) or not len(stop_idx) or not len(val_idx):
            raise ValueError(f"Empty split in fold {fold}")

        X = prepare_sequences(raw_sequences, lengths, fit_idx)
        C = prepare_context(df, fit_idx, columns)
        y_mean = float(y[fit_idx].mean())
        y_std = max(float(y[fit_idx].std()), 1.0)

        def loader(indices, shuffle=False):
            dataset = TensorDataset(
                torch.from_numpy(X[indices]),
                torch.from_numpy(lengths[indices]),
                torch.from_numpy(C[indices]),
                torch.from_numpy((y[indices] - y_mean) / y_std),
            )
            return DataLoader(dataset, batch_size=256, shuffle=shuffle)

        fit_loader = loader(fit_idx, True)
        stop_loader = loader(stop_idx)
        val_loader = loader(val_idx)
        model = DraftKingsHybridLSTM(
                sequence_features=X.shape[2],
                context_features=C.shape[1],
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        loss_fn = torch.nn.MSELoss()
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        print(f"Fold {fold}: fit={len(fit_idx):,}, stop={len(stop_idx):,}, "
              f"validate={len(val_idx):,}, context_dim={C.shape[1]}", flush=True)

        for epoch in range(1, 51):
            model.train()
            for xb, lb, cb, yb in fit_loader:
                optimizer.zero_grad()
                prediction = model(xb.to(device), lb.to(device), cb.to(device))
                loss = loss_fn(prediction, yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            stop_loss = 0.0
            with torch.no_grad():
                for xb, lb, cb, yb in stop_loader:
                    prediction = model(xb.to(device), lb.to(device), cb.to(device))
                    stop_loss += loss_fn(prediction, yb.to(device)).item() * len(yb)
            stop_loss /= len(stop_idx)
            if stop_loss < best_loss - 0.0001:
                best_loss = stop_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= 7:
                    break

        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, lb, cb, _ in val_loader:
                preds.append(model(xb.to(device), lb.to(device), cb.to(device)).cpu().numpy())
        predicted = np.maximum(np.concatenate(preds) * y_std + y_mean, 0.0)
        actual = y[val_idx]
        stats = score(actual, predicted)
        fold_metrics.append({"FOLD": fold, "BEST_EPOCH": best_epoch, **stats})
        rows = df.iloc[val_idx][["GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME"]].copy()
        rows["FOLD"] = fold
        rows["ACTUAL_DK_POINTS"] = actual
        rows["PREDICTED_DK_POINTS"] = predicted
        rows["ERROR"] = predicted - actual
        rows["HISTORY_LENGTH"] = lengths[val_idx]
        all_predictions.append(rows)
        print(f"  Best epoch {best_epoch}; MAE={stats['MAE']:.4f}, "
              f"RMSE={stats['RMSE']:.4f}, BIAS={stats['BIAS']:+.4f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    overall = score(predictions["ACTUAL_DK_POINTS"].to_numpy(),
                    predictions["PREDICTED_DK_POINTS"].to_numpy())
    pd.DataFrame([overall]).to_csv(output / "overall_metrics.csv", index=False)
    print("\nOVERALL WALK-FORWARD RESULTS")
    print(f"N: {overall['N']:,} | MAE: {overall['MAE']:.4f} | "
          f"RMSE: {overall['RMSE']:.4f} | BIAS: {overall['BIAS']:+.4f}")
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()
