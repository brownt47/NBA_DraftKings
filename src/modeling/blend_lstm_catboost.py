
"""Evaluate CatBoost and Hybrid LSTM walk-forward prediction blends."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("reports/modeling")
CATBOOST_FILE = ROOT / "boosting_comparison/predictions.csv"
LSTM_FILE = ROOT / "lstm/hybrid_full/predictions.csv"
OUTPUT_DIR = ROOT / "lstm/blending"

KEYS = ["GAME_DATE", "GAME_ID", "PLAYER_ID", "FOLD"]
WEIGHTS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(CATBOOST_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    print(f"CatBoost rows: {len(cat):,}")
    print(f"Hybrid LSTM rows: {len(lstm):,}")

    for name, df in [("CatBoost", cat), ("LSTM", lstm)]:
        duplicates = df.duplicated(KEYS).sum()
        if duplicates:
            raise ValueError(
                f"{name} contains {duplicates} duplicate player-game keys"
            )

    cat = cat[KEYS + ["ACTUAL", "CATBOOST"]].rename(
        columns={"ACTUAL": "ACTUAL_CAT"}
    )

    lstm = lstm[
        KEYS + ["ACTUAL_DK_POINTS", "PREDICTED_DK_POINTS"]
    ].rename(
        columns={
            "ACTUAL_DK_POINTS": "ACTUAL_LSTM",
            "PREDICTED_DK_POINTS": "LSTM",
        }
    )

    merged = cat.merge(
        lstm,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    print("\nMerge coverage:")
    print(merged["_merge"].value_counts().to_string())

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "Prediction files do not contain identical player-games."
        )

    merged = merged.drop(columns="_merge")

    if not np.allclose(
        merged["ACTUAL_CAT"],
        merged["ACTUAL_LSTM"],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Actual DK points differ between models.")

    merged["ACTUAL"] = merged["ACTUAL_CAT"]
    merged = merged.drop(columns=["ACTUAL_CAT", "ACTUAL_LSTM"])

    if merged[["ACTUAL", "CATBOOST", "LSTM"]].isna().any().any():
        raise ValueError("Missing actuals or predictions detected.")

    actual = merged["ACTUAL"].to_numpy()
    cat_pred = merged["CATBOOST"].to_numpy()
    lstm_pred = merged["LSTM"].to_numpy()

    cat_error = cat_pred - actual
    lstm_error = lstm_pred - actual

    correlation = np.corrcoef(cat_error, lstm_error)[0, 1]

    print(f"\nMatched observations: {len(merged):,}")
    print(f"Error correlation: {correlation:.6f}")

    results = []

    for weight in WEIGHTS:
        pred = (1 - weight) * cat_pred + weight * lstm_pred
        error = pred - actual

        results.append({
            "LSTM_WEIGHT": weight,
            "CATBOOST_WEIGHT": 1 - weight,
            "N": len(error),
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "BIAS": np.mean(error),
        })

        merged[f"BLEND_{int(weight * 100):02d}"] = pred

    summary = pd.DataFrame(results)
    baseline = summary.loc[summary["LSTM_WEIGHT"] == 0, "MAE"].iloc[0]
    summary["MAE_IMPROVEMENT"] = baseline - summary["MAE"]

    print("\nBLENDING RESULTS")
    print(summary.round(5).to_string(index=False))

    best = summary.loc[summary["MAE"].idxmin()]

    print(
        f"\nBest tested blend: "
        f"{best['CATBOOST_WEIGHT']:.0%} CatBoost / "
        f"{best['LSTM_WEIGHT']:.0%} LSTM"
    )
    print(f"Best MAE: {best['MAE']:.5f}")
    print(f"Baseline CatBoost MAE: {baseline:.5f}")

    summary.to_csv(OUTPUT_DIR / "blend_summary.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "blend_predictions.csv", index=False)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
