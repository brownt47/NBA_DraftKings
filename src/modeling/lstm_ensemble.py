
"""Evaluate CatBoost + LSTM ensembles on matched walk-forward predictions."""

from pathlib import Path

import numpy as np
import pandas as pd


BOOSTING_PATH = Path(
    "reports/modeling/boosting_comparison/predictions.csv"
)
LSTM_PATH = Path("reports/modeling/lstm/predictions.csv")
OUTPUT_DIR = Path("reports/modeling/lstm")

KEYS = ["GAME_DATE", "GAME_ID", "PLAYER_ID", "FOLD"]


def main():
    boosting = pd.read_csv(
        BOOSTING_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    lstm = pd.read_csv(
        LSTM_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    boosting["GAME_DATE"] = pd.to_datetime(boosting["GAME_DATE"])
    lstm["GAME_DATE"] = pd.to_datetime(lstm["GAME_DATE"])

    for name, frame in [("Boosting", boosting), ("LSTM", lstm)]:
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate prediction keys in {name}")

    combined = boosting.merge(
        lstm[KEYS + ["ACTUAL_DK_POINTS", "PREDICTED_DK_POINTS"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not combined["_merge"].eq("both").all():
        print(combined["_merge"].value_counts())
        raise ValueError("Prediction files do not contain identical games")

    combined = combined.drop(columns="_merge")

    if not np.allclose(
        combined["ACTUAL"],
        combined["ACTUAL_DK_POINTS"],
        atol=1e-5,
    ):
        raise ValueError("Actual DK scores differ between prediction files")

    combined = combined.rename(
        columns={"PREDICTED_DK_POINTS": "LSTM"}
    )

    actual = combined["ACTUAL"].to_numpy()
    cb = combined["CATBOOST"].to_numpy()
    lstm_pred = combined["LSTM"].to_numpy()

    def metrics(prediction):
        error = prediction - actual
        return {
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "BIAS": np.mean(error),
        }

    results = []

    for weight in np.linspace(0, 1, 11):
        prediction = (1 - weight) * cb + weight * lstm_pred

        results.append({
            "CATBOOST_WEIGHT": round(1 - weight, 2),
            "LSTM_WEIGHT": round(weight, 2),
            **metrics(prediction),
        })

    results = pd.DataFrame(results)

    error_correlation = np.corrcoef(
        cb - actual,
        lstm_pred - actual,
    )[0, 1]

    print(f"\nMatched player-games: {len(combined):,}")
    print(f"Error correlation: {error_correlation:.4f}")

    print("\nENSEMBLE RESULTS")
    print(results.round(4).to_string(index=False))

    best = results.loc[results["MAE"].idxmin()]

    print("\nBEST TESTED WEIGHT")
    print(
        f"CatBoost {best['CATBOOST_WEIGHT']:.0%} / "
        f"LSTM {best['LSTM_WEIGHT']:.0%}"
    )
    print(f"MAE: {best['MAE']:.4f}")
    print(f"RMSE: {best['RMSE']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "ensemble_weights.csv",
        index=False,
    )
    combined.to_csv(
        OUTPUT_DIR / "matched_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
