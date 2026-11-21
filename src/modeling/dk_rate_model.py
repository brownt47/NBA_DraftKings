"""Walk-forward experiment predicting NBA DK points per minute."""

from pathlib import Path
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling.lightgbm_model import (
    DEFAULT_PARAMS,
    _minutes_to_float,
    _prepare_fold_features,
)
from src.modeling.splits import biweekly_walk_forward_splits


DATA = Path("data/nba_features_2025_26.csv")
REPORT = Path("reports/modeling/dk_rate")
MINUTES_OOF = Path(
    "reports/modeling/minutes_feature_ablation/"
    "minutes_times_dk_per_min_predictions.csv"
)
DIRECT_OOF = Path(
    "reports/modeling/boosting_comparison/predictions.csv"
)

KEYS = ["GAME_ID", "PLAYER_ID"]


def metrics(actual, prediction):
    error = prediction - actual
    return {
        "N": len(error),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "BIAS": error.mean(),
        "OVER_15_PCT": 100 * (error >= 15).mean(),
        "UNDER_10_PCT": 100 * (error <= -10).mean(),
    }


def run(threshold):
    df = pd.read_csv(
        DATA,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    ).reset_index(drop=True)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["_ACTUAL_MIN"] = df["MIN"].map(_minutes_to_float)
    df["_ACTUAL_DK"] = pd.to_numeric(
        df["DK_POINTS"], errors="coerce"
    )

    df["_RATE_TARGET"] = (
        df["_ACTUAL_DK"]
        / df["_ACTUAL_MIN"].where(df["_ACTUAL_MIN"] > 0)
    )

    predictions = []

    for fold, (train_idx, val_idx) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        train = df.loc[train_idx].copy()
        val = df.loc[val_idx].copy()

        train = train.loc[
            (train["_ACTUAL_MIN"] >= threshold)
            & np.isfinite(train["_RATE_TARGET"])
        ].copy()

        X_train, X_val = _prepare_fold_features(train, val)

        forbidden = {
            "MIN", "DK_POINTS", "_ACTUAL_MIN",
            "_ACTUAL_DK", "_RATE_TARGET",
        }

        leaked = forbidden.intersection(X_train.columns)
        if leaked:
            raise ValueError(f"Leaked predictors: {leaked}")

        model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        model.fit(X_train, train["_RATE_TARGET"])

        rate = np.maximum(model.predict(X_val), 0)

        result = val[
            KEYS + ["GAME_DATE", "_ACTUAL_DK"]
        ].copy()

        result["FOLD"] = fold
        result["PREDICTED_RATE"] = rate
        predictions.append(result)

        print(
            f"Fold {fold}: "
            f"train={len(train):,}, "
            f"validation={len(val):,}",
            flush=True,
        )

    oof = pd.concat(predictions, ignore_index=True)

    minutes = pd.read_csv(
        MINUTES_OOF,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    direct = pd.read_csv(
        DIRECT_OOF,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    oof = oof.merge(
        minutes[
            KEYS + [
                "FOLD", "ROLE", "D_PARTICIPATION",
                "DK_PER_MIN", "DK_POINTS",
            ]
        ],
        on=KEYS,
        suffixes=("", "_MIN"),
        validate="one_to_one",
    )

    oof = oof.merge(
        direct[KEYS + ["FOLD", "CATBOOST"]],
        on=KEYS,
        suffixes=("", "_DIRECT"),
        validate="one_to_one",
    )

    if not (
        (oof["FOLD"] == oof["FOLD_MIN"]).all()
        and (oof["FOLD"] == oof["FOLD_DIRECT"]).all()
    ):
        raise ValueError("Fold assignments do not match")

    if not np.allclose(
        oof["_ACTUAL_DK"], oof["DK_POINTS"]
    ):
        raise ValueError("Actual DK scores do not match")

    oof = oof.replace([np.inf, -np.inf], np.nan)
    oof = oof.dropna(
        subset=[
            "PREDICTED_RATE", "D_PARTICIPATION",
            "DK_PER_MIN", "CATBOOST", "_ACTUAL_DK",
        ]
    )
    oof = oof.loc[oof["DK_PER_MIN"] > 0].copy()

    minutes_pred = oof["D_PARTICIPATION"].clip(lower=0)

    oof["HISTORICAL_HYBRID"] = (
        minutes_pred * oof["DK_PER_MIN"]
    )

    oof["LEARNED_HYBRID"] = (
        minutes_pred * oof["PREDICTED_RATE"]
    )

    oof["SHRUNK_HYBRID"] = (
        0.5 * oof["LEARNED_HYBRID"]
        + 0.5 * oof["HISTORICAL_HYBRID"]
    )

    models = [
        "CATBOOST",
        "HISTORICAL_HYBRID",
        "LEARNED_HYBRID",
        "SHRUNK_HYBRID",
    ]

    for name in [
        "HISTORICAL_HYBRID",
        "LEARNED_HYBRID",
        "SHRUNK_HYBRID",
    ]:
        blend = "BLEND_75_" + name
        oof[blend] = (
            0.75 * oof["CATBOOST"]
            + 0.25 * oof[name]
        )
        models.append(blend)

    rows = []

    for name in models:
        result = metrics(oof["_ACTUAL_DK"], oof[name])
        result["MODEL"] = name
        rows.append(result)

    summary = pd.DataFrame(rows).sort_values("MAE")

    REPORT.mkdir(parents=True, exist_ok=True)

    oof.to_csv(
        REPORT / f"predictions_min{threshold}.csv",
        index=False,
    )

    summary.to_csv(
        REPORT / f"summary_min{threshold}.csv",
        index=False,
    )

    print("\nRESULTS")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved results to {REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-train-minutes",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    run(args.min_train_minutes)
