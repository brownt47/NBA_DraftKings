
"""Single-fold GPU experiment for the DraftKings LSTM."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.model_adapters.lstm_adapter import (
    load_historical_sequences,
)
from src.modeling.lstm_model import DraftKingsLSTM
from src.modeling.splits import biweekly_walk_forward_splits


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df, X, lengths = load_historical_sequences()
    y = df["DK_POINTS"].to_numpy(dtype=np.float32)

    train_idx, val_idx = next(
        biweekly_walk_forward_splits(df)
    )

    dates = pd.to_datetime(df["GAME_DATE"])
    unique_dates = np.sort(dates.iloc[train_idx].unique())
    split_at = max(1, int(len(unique_dates) * 0.9))
    cutoff = unique_dates[split_at - 1]

    fit_idx = train_idx[
        dates.iloc[train_idx].to_numpy() <= cutoff
    ]
    stop_idx = train_idx[
        dates.iloc[train_idx].to_numpy() > cutoff
    ]

    if not len(stop_idx):
        raise ValueError("Empty early-stopping set")

    # Scale using real historical observations from the
    # fitting period, excluding padded positions.
    valid = (
        np.arange(X.shape[1])[None, :]
        >= X.shape[1] - lengths[fit_idx, None]
    )

    observations = X[fit_idx][valid]
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    X = np.nan_to_num(
        (X - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # Restore padding to zero after scaling.
    for i in range(len(X)):
        X[i, :X.shape[1] - lengths[i]] = 0

    # Standardize the target using training labels only.
    y_mean = float(y[fit_idx].mean())
    y_std = float(y[fit_idx].std())
    y_std = max(y_std, 1.0)

    def make_loader(indices, shuffle=False):
        dataset = TensorDataset(
            torch.from_numpy(X[indices]),
            torch.from_numpy(lengths[indices]),
            torch.from_numpy(
                (y[indices] - y_mean) / y_std
            ),
        )
        return DataLoader(
            dataset,
            batch_size=256,
            shuffle=shuffle,
        )

    train_loader = make_loader(fit_idx, True)
    stop_loader = make_loader(stop_idx)
    val_loader = make_loader(val_idx)

    model = DraftKingsLSTM(
        input_size=X.shape[2]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001,
    )
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 7
    stale = 0

    print(f"Training rows: {len(fit_idx):,}")
    print(f"Early stopping rows: {len(stop_idx):,}")
    print(f"Validation rows: {len(val_idx):,}")

    for epoch in range(1, 51):
        model.train()
        total = 0.0

        for xb, lb, yb in train_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            predictions = model(xb, lb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            total += loss.item() * len(yb)

        model.eval()
        stop_total = 0.0

        with torch.no_grad():
            for xb, lb, yb in stop_loader:
                predictions = model(
                    xb.to(device), lb.to(device)
                )
                stop_total += (
                    loss_fn(
                        predictions, yb.to(device)
                    ).item() * len(yb)
                )

        train_loss = total / len(fit_idx)
        stop_loss = stop_total / len(stop_idx)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Stop MSE: {stop_loss:.4f}"
        )

        if stop_loss < best_loss - 0.0001:
            best_loss = stop_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_state)
    model.eval()

    predicted = []
    with torch.no_grad():
        for xb, lb, _ in val_loader:
            output = model(
                xb.to(device), lb.to(device)
            )
            predicted.extend(output.cpu().numpy())

    predicted = np.maximum(
        np.asarray(predicted) * y_std + y_mean,
        0,
    )

    actual = y[val_idx]
    error = predicted - actual

    print("\nFIRST FOLD RESULTS")
    print("N:", len(actual))
    print(f"MAE:  {np.abs(error).mean():.4f}")
    print(f"RMSE: {np.sqrt(np.mean(error**2)):.4f}")
    print(f"BIAS: {error.mean():+.4f}")


if __name__ == "__main__":
    main()
