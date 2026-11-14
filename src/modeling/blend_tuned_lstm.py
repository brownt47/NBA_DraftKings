
"""Compare CatBoost with the tuned Hybrid LSTM using walk-forward predictions."""

from pathlib import Path

import numpy as np
import pandas as pd


CATBOOST_FILE = Path(
    "reports/modeling/boosting_comparison/predictions.csv"
)
LSTM_FILE = Path(
    "reports/modeling/lstm/tuned_full_seq10/predictions.csv"
)
OUTPUT_DIR = Path("reports/modeling/lstm/tuned_blending")

KEYS = ["GAME_DATE", "GAME_ID", "PLAYER_ID", "FOLD"]


def metrics(actual, predicted):
    errors = predicted - actual
    return {
        "N": len(actual),
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "BIAS": float(errors.mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cb = pd.read_csv(CATBOOST_FILE)
    lstm = pd.read_csv(LSTM_FILE)

    # Require unique player-games in both files.
    for name, df in [("CatBoost", cb), ("LSTM", lstm)]:
        if df.duplicated(KEYS).any():
            raise ValueError(f"Duplicate prediction keys in {name}")

    merged = cb.merge(
        lstm[
            KEYS + [
                "ACTUAL_DK_POINTS",
                "PREDICTED_DK_POINTS",
                "HISTORY_LENGTH",
            ]
        ],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(cb) or len(merged) != len(lstm):
        raise ValueError(
            f"Unmatched predictions: CatBoost={len(cb)}, "
            f"LSTM={len(lstm)}, matched={len(merged)}"
        )

    if not np.allclose(
        merged["ACTUAL"],
        merged["ACTUAL_DK_POINTS"],
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("Actual DK points differ between models")

    merged = merged.sort_values(KEYS).reset_index(drop=True)

    actual = merged["ACTUAL"].to_numpy(dtype=float)
    catboost = merged["CATBOOST"].to_numpy(dtype=float)
    lstm_pred = merged["PREDICTED_DK_POINTS"].to_numpy(dtype=float)

    if not all(
        np.isfinite(values).all()
        for values in [actual, catboost, lstm_pred]
    ):
        raise ValueError("Non-finite actuals or predictions")

    correlation = np.corrcoef(
        catboost - actual,
        lstm_pred - actual,
    )[0, 1]

    print(f"Matched predictions: {len(merged):,}")
    print(f"Error correlation: {correlation:.6f}")

    summary_rows = []
    fold_rows = []

    for weight in np.linspace(0, 0.50, 11):
        predicted = (1 - weight) * catboost + weight * lstm_pred

        summary_rows.append({
            "LSTM_WEIGHT": round(float(weight), 2),
            "CATBOOST_WEIGHT": round(float(1 - weight), 2),
            **metrics(actual, predicted),
        })

        for fold, group in merged.groupby("FOLD"):
            indices = group.index.to_numpy()

            fold_rows.append({
                "FOLD": int(fold),
                "LSTM_WEIGHT": round(float(weight), 2),
                **metrics(actual[indices], predicted[indices]),
            })

    summary = pd.DataFrame(summary_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    best = summary.loc[summary["MAE"].idxmin()]
    best_weight = float(best["LSTM_WEIGHT"])

    merged["BLENDED_PREDICTION"] = (
        (1 - best_weight) * catboost + best_weight * lstm_pred
    )
    merged["BLENDED_ERROR"] = merged["BLENDED_PREDICTION"] - actual

    summary.to_csv(OUTPUT_DIR / "weight_comparison.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_comparison.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "blend_predictions.csv", index=False)

    print("\nBLEND WEIGHT COMPARISON")
    print(summary.to_string(index=False, float_format="%.4f"))

    print("\nBEST BLEND")
    print(
        f"CatBoost: {1 - best_weight:.0%} | "
        f"LSTM: {best_weight:.0%}"
    )
    print(
        f"MAE: {best['MAE']:.4f} | "
        f"RMSE: {best['RMSE']:.4f} | "
        f"BIAS: {best['BIAS']:+.4f}"
    )

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
