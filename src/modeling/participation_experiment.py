"""Compare baseline CatBoost against participation-enhanced CatBoost.

Run from the repository root:
    python -m src.modeling.participation_experiment

Uses January-March 2026 biweekly walk-forward validation. April remains held out.
Does not change production features or models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.features import PREGAME_FEATURES
from src.modeling.splits import biweekly_walk_forward_splits

DATA_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/participation")
CATEGORICAL = ["OPPONENT", "HOME_AWAY"]

PARTICIPATION_FEATURES = [
    "TEAM_GAMES_SINCE_LAST_APPEARANCE",
    "DAYS_SINCE_LAST_APPEARANCE",
    "MIN_PER_TEAM_GAME_LAST_5",
    "MIN_PER_TEAM_GAME_LAST_10",
    "MINUTES_KNOWN_LAST_5",
    "MINUTES_KNOWN_LAST_10",
    "APPEARANCE_RATE_LAST_5",
    "APPEARANCE_RATE_LAST_10",
    "PARTICIPATION_KNOWN_LAST_5",
    "PARTICIPATION_KNOWN_LAST_10",
    "COACH_DNP_LAST_5",
    "COACH_DNP_LAST_10",
    "PARTICIPATION_COVERAGE_LAST_5",
    "PARTICIPATION_COVERAGE_LAST_10",
]

FEATURE_SETS = {
    "BASELINE": list(PREGAME_FEATURES),
    "PARTICIPATION": list(dict.fromkeys([*PREGAME_FEATURES, *PARTICIPATION_FEATURES])),
}

MODEL_PARAMS = dict(
    iterations=500,
    depth=7,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    thread_count=-1,
    verbose=False,
    allow_writing_files=False,
)


def prepare_features(df, columns):
    X = df[columns].copy()
    for col in columns:
        if col in CATEGORICAL:
            X[col] = X[col].fillna("UNKNOWN").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def metrics(df, prediction_col):
    error = df[prediction_col] - df["ACTUAL"]
    return {
        "N": len(df),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "BIAS": error.mean(),
        "OVER_10_PCT": 100 * (error >= 10).mean(),
        "OVER_15_PCT": 100 * (error >= 15).mean(),
    }


def compare(group, label):
    if group.empty:
        return None
    a = metrics(group, "BASELINE")
    b = metrics(group, "PARTICIPATION")
    return {
        "SEGMENT": label,
        "N": len(group),
        "BASELINE_MAE": a["MAE"],
        "PARTICIPATION_MAE": b["MAE"],
        "MAE_IMPROVEMENT": a["MAE"] - b["MAE"],
        "BASELINE_OVER_10_PCT": a["OVER_10_PCT"],
        "PARTICIPATION_OVER_10_PCT": b["OVER_10_PCT"],
    }


def run():
    df = pd.read_csv(DATA_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}, low_memory=False)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    required = set(FEATURE_SETS["PARTICIPATION"]) | {"DK_POINTS", "GAME_DATE"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df):,} player-games, {len(df.columns)} columns")
    for name, columns in FEATURE_SETS.items():
        print(f"{name}: {len(columns)} predictors")

    predictions = []
    for fold, (train_idx, val_idx) in enumerate(biweekly_walk_forward_splits(df), start=1):
        train = df.loc[train_idx].dropna(subset=["DK_POINTS"]).copy()
        val = df.loc[val_idx].dropna(subset=["DK_POINTS"]).copy()
        metadata = [
            c for c in [
                "GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM",
                "MIN", "MIN_AVG_LAST_5", "MIN_STD_LAST_5", "MIN_STD_LAST_10",
                "APPEARANCE_RATE_LAST_5", "MIN_PER_TEAM_GAME_LAST_5",
                "COACH_DNP_LAST_5", "PARTICIPATION_COVERAGE_LAST_5",
            ] if c in val.columns
        ]
        result = val[metadata].copy()
        result["FOLD"] = fold
        result["ACTUAL"] = val["DK_POINTS"]

        for name, columns in FEATURE_SETS.items():
            model = CatBoostRegressor(**MODEL_PARAMS)
            model.fit(
                prepare_features(train, columns),
                train["DK_POINTS"],
                cat_features=CATEGORICAL,
            )
            result[name] = np.maximum(model.predict(prepare_features(val, columns)), 0)

        predictions.append(result)
        print(f"Fold {fold}: train={len(train):,}, validation={len(val):,}, "
              f"baseline MAE={metrics(result, 'BASELINE')['MAE']:.4f}, "
              f"participation MAE={metrics(result, 'PARTICIPATION')['MAE']:.4f}", flush=True)

    combined = pd.concat(predictions, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    overall = pd.DataFrame({name: metrics(combined, name) for name in FEATURE_SETS}).T
    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv")
    print("\nOVERALL RESULTS")
    print(overall.round(4).to_string())

    segments = [compare(combined, "ALL")]
    if "MIN_AVG_LAST_5" in combined:
        mins = pd.to_numeric(combined["MIN_AVG_LAST_5"], errors="coerce")
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, float("inf"))]:
            segments.append(compare(combined.loc[(mins >= lo) & (mins < hi)],
                                    f"PREGAME_MIN_AVG_{lo}_TO_{hi}"))
    if "MIN_STD_LAST_5" in combined:
        std = pd.to_numeric(combined["MIN_STD_LAST_5"], errors="coerce")
        for threshold in (5, 8, 10):
            segments.append(compare(combined.loc[std >= threshold],
                                    f"PREGAME_MIN_STD_5_GE_{threshold}"))
    if "APPEARANCE_RATE_LAST_5" in combined:
        rate = pd.to_numeric(combined["APPEARANCE_RATE_LAST_5"], errors="coerce")
        segments.append(compare(combined.loc[rate < 0.8], "PREGAME_APPEARANCE_RATE_LT_0.8"))
        segments.append(compare(combined.loc[rate >= 0.8], "PREGAME_APPEARANCE_RATE_GE_0.8"))

    segment_df = pd.DataFrame([s for s in segments if s is not None])
    segment_df.to_csv(OUTPUT_DIR / "segment_metrics.csv", index=False)
    print("\nSEGMENT RESULTS (pregame-defined groups)")
    print(segment_df.round(3).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
