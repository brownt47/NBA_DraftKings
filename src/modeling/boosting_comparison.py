"""Walk-forward comparison of LightGBM, CatBoost, and their ensemble."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.lightgbm_model import (
    DEFAULT_PARAMS,
    _prepare_fold_features,
)
from src.modeling.splits import biweekly_walk_forward_splits

DATA_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/boosting_comparison")
CATEGORICAL = ["OPPONENT", "HOME_AWAY"]


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

        X_train, X_val = _prepare_fold_features(train, val)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        lgb_model.fit(X_train, train["DK_POINTS"])
        lgb_pred = np.maximum(lgb_model.predict(X_val), 0)

        # CatBoost uses string categorical predictors.
        cb_train = X_train.copy()
        cb_val = X_val.copy()

        for col in CATEGORICAL:
            cb_train[col] = (
                cb_train[col].astype(object)
                .fillna("UNKNOWN").astype(str)
            )
            cb_val[col] = (
                cb_val[col].astype(object)
                .fillna("UNKNOWN").astype(str)
            )

        cb_model = CatBoostRegressor(
            iterations=500,
            depth=7,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )

        cb_model.fit(
            cb_train,
            train["DK_POINTS"],
            cat_features=CATEGORICAL,
        )

        cb_pred = np.maximum(cb_model.predict(cb_val), 0)

        result = val[[
            "GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME"
        ]].copy()

        result["FOLD"] = fold
        result["ACTUAL"] = val["DK_POINTS"]
        result["LIGHTGBM"] = lgb_pred
        result["CATBOOST"] = cb_pred
        result["ENSEMBLE_50_50"] = 0.5 * (lgb_pred + cb_pred)

        predictions.append(result)

        print(
            f"Fold {fold}: "
            f"{len(train):,} training, "
            f"{len(val):,} validation"
        )

    combined = pd.concat(predictions, ignore_index=True)

    overall = pd.DataFrame({
        name: metrics(combined, name)
        for name in ["LIGHTGBM", "CATBOOST", "ENSEMBLE_50_50"]
    }).T

    fold_results = []

    for fold, group in combined.groupby("FOLD"):
        for name in ["LIGHTGBM", "CATBOOST", "ENSEMBLE_50_50"]:
            fold_results.append({
                "FOLD": fold,
                "MODEL": name,
                **metrics(group, name),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        OUTPUT_DIR / "predictions.csv", index=False
    )
    overall.to_csv(
        OUTPUT_DIR / "overall_metrics.csv"
    )
    pd.DataFrame(fold_results).to_csv(
        OUTPUT_DIR / "fold_metrics.csv", index=False
    )

    print("\nOVERALL RESULTS")
    print(overall.round(4).to_string())

    print("\nFOLD RESULTS")
    print(
        pd.DataFrame(fold_results)
        .pivot(index="FOLD", columns="MODEL", values="MAE")
        .round(4).to_string()
    )


if __name__ == "__main__":
    run()
