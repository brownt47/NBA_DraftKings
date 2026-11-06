"""Historical walk-forward XGBoost comparison against saved models."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.splits import biweekly_walk_forward_splits
from src.modeling.xgboost_model import train_xgboost


DATA_PATH = Path("data/nba_features_2025_26.csv")
BENCHMARK_PATH = Path(
    "reports/modeling/boosting_comparison/predictions.csv"
)
OUTPUT_DIR = Path("reports/modeling/xgboost_comparison")

KEYS = ["FOLD", "GAME_ID", "PLAYER_ID"]
MODELS = [
    "LIGHTGBM",
    "CATBOOST",
    "ENSEMBLE_50_50",
    "XGBOOST",
]


def metrics(df, column):
    error = df[column] - df["ACTUAL"]

    return {
        "N": len(df),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "BIAS": error.mean(),
    }


def run():
    df = pd.read_csv(
        DATA_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    benchmark = pd.read_csv(
        BENCHMARK_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    required = KEYS + ["ACTUAL"] + MODELS[:-1]
    missing = [c for c in required if c not in benchmark.columns]

    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    if benchmark.duplicated(KEYS).any():
        raise ValueError("Duplicate benchmark prediction keys")

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        train = df.loc[train_idx].dropna(
            subset=["DK_POINTS"]
        ).copy()

        val = df.loc[val_idx].dropna(
            subset=["DK_POINTS"]
        ).copy()

        model, predicted = train_xgboost(train, val)

        result = val[
            ["GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME"]
        ].copy()

        result["FOLD"] = fold
        result["ACTUAL"] = val["DK_POINTS"].to_numpy()
        result["XGBOOST"] = predicted

        predictions.append(result)

        print(
            f"Fold {fold}: "
            f"{len(train):,} training, "
            f"{len(val):,} validation"
        )

    combined = pd.concat(predictions, ignore_index=True)

    if combined.duplicated(KEYS).any():
        raise ValueError("Duplicate XGBoost prediction keys")

    if not np.isfinite(combined["XGBOOST"]).all():
        raise ValueError("Nonfinite XGBoost predictions")

    comparison = benchmark.merge(
        combined[KEYS + ["ACTUAL", "XGBOOST"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_XGB"),
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "XGBoost evaluation population differs from benchmark"
        )

    if not np.allclose(
        comparison["ACTUAL"],
        comparison["ACTUAL_XGB"],
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError("Actual fantasy points differ from benchmark")

    comparison = comparison.drop(
        columns=["ACTUAL_XGB", "_merge"]
    )

    overall = pd.DataFrame({
        name: metrics(comparison, name)
        for name in MODELS
    }).T

    fold_results = []

    for fold, group in comparison.groupby("FOLD"):
        for name in MODELS:
            fold_results.append({
                "FOLD": fold,
                "MODEL": name,
                **metrics(group, name),
            })

    win_results = []

    xgb_error = (
        comparison["XGBOOST"] - comparison["ACTUAL"]
    ).abs()

    for opponent in MODELS[:-1]:
        other_error = (
            comparison[opponent] - comparison["ACTUAL"]
        ).abs()

        win_results.append({
            "OPPONENT": opponent,
            "XGBOOST_WIN_RATE": (xgb_error < other_error).mean(),
            "OPPONENT_WIN_RATE": (other_error < xgb_error).mean(),
            "TIE_RATE": (xgb_error == other_error).mean(),
            "MAE_DIFFERENCE": (
                xgb_error.mean() - other_error.mean()
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUT_DIR / "predictions.csv", index=False
    )

    overall.to_csv(
        OUTPUT_DIR / "overall_metrics.csv"
    )

    pd.DataFrame(fold_results).to_csv(
        OUTPUT_DIR / "fold_metrics.csv", index=False
    )

    pd.DataFrame(win_results).to_csv(
        OUTPUT_DIR / "head_to_head.csv", index=False
    )

    print("\nOVERALL RESULTS")
    print(overall.round(4).to_string())

    print("\nFOLD MAE")
    print(
        pd.DataFrame(fold_results)
        .pivot(index="FOLD", columns="MODEL", values="MAE")
        .round(4).to_string()
    )

    print("\nXGBOOST HEAD-TO-HEAD")
    print(
        pd.DataFrame(win_results).round(4).to_string(index=False)
    )

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
