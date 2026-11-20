"""Historical comparison of rolling-average and CatBoost minutes floors."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dk_features import load_player_games, load_team_games
from src.pregame import build_pregame_features
from src.modeling.model_registry import get_model_adapter
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path("data/dk_backtest")
OUTPUT = ROOT / "predicted_minutes"
CACHE = OUTPUT / "minutes_predictions"
MIN_SALARY = 3500
MAX_TEAM = 3

CONFIGS = [
    ("LAST5_22", "MIN_AVG_LAST_5", 22),
    ("LAST5_23", "MIN_AVG_LAST_5", 23),
    ("CATBOOST_22", "PRED_MIN", 22),
    ("CATBOOST_23", "PRED_MIN", 23),
]

FEATURE_FILES = [
    Path("data/nba_features_2025_26.csv"),
    Path("data/nba_features_2026_27.csv"),
]


def projection_files():
    paths = []
    for folder in [
        ROOT / "projections",
        ROOT / "extended" / "projections",
    ]:
        paths.extend(folder.glob("projections_*.pkl"))
    return sorted(paths)


def minutes_predictions(date, slate, history, teams, training):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"minutes_{date.replace('-', '_')}.csv"

    if path.exists():
        cached = pd.read_csv(path, dtype={"PLAYER_ID": str})
        if cached["PLAYER_ID"].duplicated().any():
            raise ValueError(f"Duplicate cached player IDs: {date}")
        return cached

    print(f"  Building pregame features for {date}", flush=True)

    # The archived projection slate already contains mapped ESPN IDs.
    mapped = slate.copy()

    # Recreate the fields expected by build_pregame_features.
    mapped["ESPN_PLAYER_ID"] = mapped["ESPN_PLAYER_ID"].astype(str)

    features = build_pregame_features(mapped, history, teams)

    adapter = get_model_adapter("catboost")

    historical = training.loc[
        training["GAME_DATE"] < pd.Timestamp(date)
    ].copy()

    X_train, prep = adapter.prepare_training_features(historical)
    X_predict, _ = adapter.prepare_training_features(features)

    X_predict = X_predict.reindex(columns=X_train.columns)

    y = pd.to_numeric(historical["MIN"], errors="coerce")
    valid = y.notna()

    print(
        f"  Training minutes model: {valid.sum():,} games",
        flush=True,
    )

    model = adapter.create_model()
    model.fit(
        X_train.loc[valid],
        y.loc[valid],
        cat_features=prep["categories"],
    )

    predicted = np.maximum(model.predict(X_predict), 0)

    result = pd.DataFrame({
        "PLAYER_ID": features["PLAYER_ID"].astype(str).to_numpy(),
        "PRED_MIN": predicted,
    })

    if result["PLAYER_ID"].duplicated().any():
        raise ValueError(f"Duplicate predicted player IDs: {date}")

    result.to_csv(path, index=False)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    training = pd.concat(
        [pd.read_csv(path) for path in FEATURE_FILES],
        ignore_index=True,
    )
    training["GAME_DATE"] = pd.to_datetime(training["GAME_DATE"])

    history = load_player_games()
    teams = load_team_games()

    records = []
    lineup_records = []

    for path in projection_files():
        date = path.stem.replace("projections_", "").replace("_", "-")
        print(f"\n=== {date} ===", flush=True)

        slate = pd.read_pickle(path).copy()
        slate["PLAYER_ID"] = slate["PLAYER_ID"].astype(str)

        predicted = minutes_predictions(
            date, slate, history, teams, training
        )

        slate = slate.merge(
            predicted,
            on="PLAYER_ID",
            how="left",
            validate="one_to_one",
        )

        actual = load_actual_results(history, date)

        for config, source, floor in CONFIGS:
            test_slate = slate.copy()

            # Preserve the historical rolling average for diagnostics.
            test_slate["ORIGINAL_MIN_AVG_LAST_5"] = (
                test_slate["MIN_AVG_LAST_5"]
            )

            if source == "PRED_MIN":
                test_slate["MIN_AVG_LAST_5"] = test_slate["PRED_MIN"]

            try:
                result = optimize_classic(
                    test_slate,
                    min_salary=MIN_SALARY,
                    min_minutes=floor,
                    max_players_per_team=MAX_TEAM,
                )

                lineup = result["lineup"].copy()
                lineup["PLAYER_ID"] = lineup["PLAYER_ID"].astype(str)

                lineup = lineup.merge(
                    actual,
                    on="PLAYER_ID",
                    how="left",
                    validate="one_to_one",
                )

                missing = int(lineup["ACTUAL_DK_POINTS"].isna().sum())
                actual_score = (
                    float(lineup["ACTUAL_DK_POINTS"].sum())
                    if missing == 0 else np.nan
                )

                records.append({
                    "DATE": date,
                    "CONFIG": config,
                    "STATUS": "COMPLETE" if missing == 0 else "MISSING_RESULTS",
                    "PROJECTED": result["projected_points"],
                    "ACTUAL": actual_score,
                    "SALARY": result["total_salary"],
                    "MISSING_RESULTS": missing,
                    "LINEUP_KEY": "|".join(
                        sorted(lineup["DK_ID"].astype(str))
                    ),
                })

                lineup["DATE"] = date
                lineup["CONFIG"] = config
                lineup_records.append(lineup)

                print(
                    f"  {config:12} "
                    f"Projected={result['projected_points']:7.2f} "
                    f"Actual={actual_score:7.2f} "
                    f"Missing={missing}",
                    flush=True,
                )

            except (ValueError, RuntimeError) as exc:
                records.append({
                    "DATE": date,
                    "CONFIG": config,
                    "STATUS": "FAILED",
                    "ERROR": str(exc),
                })
                print(f"  {config}: FAILED ({exc})", flush=True)

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    if lineup_records:
        pd.concat(lineup_records, ignore_index=True).to_csv(
            OUTPUT / "lineups.csv", index=False
        )

    complete = summary.loc[summary["STATUS"] == "COMPLETE"]

    print("\n=== OVERALL RESULTS ===")

    if not complete.empty:
        comparison = complete.groupby("CONFIG").agg(
            SLATES=("DATE", "nunique"),
            AVG_ACTUAL=("ACTUAL", "mean"),
            AVG_PROJECTED=("PROJECTED", "mean"),
        ).round(2)

        print(comparison.to_string())
        comparison.to_csv(OUTPUT / "comparison.csv")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
