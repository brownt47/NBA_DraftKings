"""Seven-fold walk-forward evaluation of the historical-game DraftKings LSTM.

Run: python -m src.modeling.lstm_walkforward
"""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.lstm_model import DraftKingsLSTM
from src.modeling.model_adapters.lstm_adapter import load_historical_sequences
from src.modeling.splits import biweekly_walk_forward_splits


OUTPUT_DIR = Path("reports/modeling/lstm")
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 7


def run_fold(df, raw_x, lengths, y, train_idx, val_idx, fold):
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)
    random.seed(42 + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dates = pd.to_datetime(df["GAME_DATE"])
    train_dates = np.sort(dates.iloc[train_idx].unique())
    split_at = max(1, int(len(train_dates) * 0.9))
    cutoff = train_dates[split_at - 1]
    train_mask = dates.iloc[train_idx].to_numpy() <= cutoff
    fit_idx = train_idx[train_mask]
    stop_idx = train_idx[~train_mask]
    if not len(fit_idx) or not len(stop_idx) or not len(val_idx):
        raise ValueError(f"Fold {fold}: empty fit, stopping, or validation split")

    # Fit scaler exclusively on historical observations belonging to fitting rows.
    valid = (np.arange(raw_x.shape[1])[None, :]
             >= raw_x.shape[1] - lengths[fit_idx, None])
    observed = raw_x[fit_idx][valid]
    if not len(observed):
        raise ValueError("No historical observations for scaling")
    mean = observed.mean(axis=0)
    std = observed.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    # Only scale rows used in this fold to avoid holding a redundant full copy.
    selected = np.concatenate([fit_idx, stop_idx, val_idx])
    x = np.nan_to_num((raw_x[selected] - mean) / std,
                      nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seq_len = x.shape[1]
    padding = np.arange(seq_len)[None, :] < seq_len - lengths[selected, None]
    x[padding] = 0.0
    offsets = {int(idx): i for i, idx in enumerate(selected)}

    y_mean = float(y[fit_idx].mean())
    y_std = max(float(y[fit_idx].std()), 1.0)

    def loader(indices, shuffle=False):
        local = np.array([offsets[int(i)] for i in indices], dtype=np.int64)
        data = TensorDataset(
            torch.from_numpy(x[local]),
            torch.from_numpy(lengths[indices]),
            torch.from_numpy(((y[indices] - y_mean) / y_std).astype(np.float32)),
        )
        return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)

    fit_loader = loader(fit_idx, True)
    stop_loader = loader(stop_idx)
    val_loader = loader(val_idx)
    model = DraftKingsLSTM(input_size=raw_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    print(f"Fold {fold}: fit={len(fit_idx):,}, stop={len(stop_idx):,}, "
          f"validate={len(val_idx):,}", flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, lb, yb in fit_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        stop_sum = 0.0
        with torch.no_grad():
            for xb, lb, yb in stop_loader:
                pred = model(xb.to(device), lb.to(device))
                stop_sum += criterion(pred, yb.to(device)).item() * len(yb)
        stop_loss = stop_sum / len(stop_idx)
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
    predicted = []
    with torch.no_grad():
        for xb, lb, _ in val_loader:
            predicted.append(model(xb.to(device), lb.to(device)).cpu().numpy())
    predictions = np.maximum(np.concatenate(predicted) * y_std + y_mean, 0)
    actual = y[val_idx]
    errors = predictions - actual
    metrics = {
        "FOLD": fold,
        "START_DATE": str(dates.iloc[val_idx].min().date()),
        "END_DATE": str(dates.iloc[val_idx].max().date()),
        "N": len(val_idx),
        "BEST_EPOCH": best_epoch,
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "BIAS": float(errors.mean()),
    }
    print(f"  Best epoch {best_epoch}; MAE={metrics['MAE']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, BIAS={metrics['BIAS']:+.4f}", flush=True)

    identifiers = [c for c in ["GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM"]
                   if c in df.columns]
    results = df.iloc[val_idx][identifiers].copy()
    results["FOLD"] = fold
    results["ACTUAL_DK_POINTS"] = actual
    results["PREDICTED_DK_POINTS"] = predictions
    results["ERROR"] = errors
    results["HISTORY_LENGTH"] = lengths[val_idx]
    return metrics, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    df, raw_x, lengths = load_historical_sequences()
    y = pd.to_numeric(df["DK_POINTS"], errors="raise").to_numpy(dtype=np.float32)
    metrics, predictions = [], []
    for fold, (train_idx, val_idx) in enumerate(biweekly_walk_forward_splits(df), start=1):
        if not len(train_idx) or not len(val_idx):
            continue
        m, p = run_fold(df, raw_x, lengths, y, train_idx, val_idx, fold)
        metrics.append(m)
        predictions.append(p)
    if not metrics:
        raise RuntimeError("No valid walk-forward folds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    error = pred_df["ERROR"].to_numpy()
    print("\nOVERALL WALK-FORWARD RESULTS")
    print(f"Folds: {len(metrics_df)} | N: {len(error):,}")
    print(f"MAE: {np.abs(error).mean():.4f}")
    print(f"RMSE: {np.sqrt(np.mean(error ** 2)):.4f}")
    print(f"BIAS: {error.mean():+.4f}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
