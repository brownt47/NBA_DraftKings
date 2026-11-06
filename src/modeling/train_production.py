
"""Train and save a LightGBM model for upcoming NBA games."""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.modeling.features import prepare_model_features
from src.modeling.lightgbm_model import DEFAULT_PARAMS


DEFAULT_DATA = [
    Path("data/nba_features_2025_26.csv"),
    Path("data/nba_features_2026_27.csv"),
]
DEFAULT_MODEL = Path("models/lightgbm_production.joblib")


def train_production_model(data_path, model_path, cutoff_date):
    """Train using completed games strictly before cutoff_date."""

    data_paths = (
        [data_path]
        if isinstance(data_path, (str, Path))
        else list(data_path)
    )

    frames = []

    for path in data_paths:
        path = Path(path)
        df_season = pd.read_csv(path)

        print(f"Loaded {len(df_season):,} rows from {path}")

        frames.append(df_season)

    df = pd.concat(frames, ignore_index=True)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["DK_POINTS"] = pd.to_numeric(
        df["DK_POINTS"], errors="coerce"
    )

    cutoff = pd.Timestamp(cutoff_date)

    df = df.loc[
        (df["GAME_DATE"] < cutoff)
        & df["DK_POINTS"].notna()
    ].copy()

    if df.empty:
        raise ValueError("No eligible historical games found.")

    # Prevent accidentally training on duplicate player-games.
    if {"GAME_ID", "PLAYER_ID"}.issubset(df.columns):
        duplicates = df.duplicated(
            subset=["GAME_ID", "PLAYER_ID"]
        )

        if duplicates.any():
            raise ValueError(
                f"Found {duplicates.sum()} duplicate player-game rows."
            )


    X = prepare_model_features(df)
    y = pd.to_numeric(df["DK_POINTS"], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": list(X.columns),
        "categories": {
            column: list(X[column].cat.categories)
            for column in ["OPPONENT", "HOME_AWAY"]
        },
        "cutoff_date": str(cutoff.date()),
        "training_rows": len(X),
    }

    joblib.dump(artifact, model_path)

    print(f"Training records: {len(X):,}")
    print(f"Predictors: {len(X.columns)}")
    print(f"Cutoff date: {cutoff.date()}")
    print(f"Saved model: {model_path}")

    return artifact


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=DEFAULT_DATA,
        help="One or more historical feature CSV files.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cutoff", required=True)

    args = parser.parse_args()

    train_production_model(args.data, args.model, args.cutoff)


if __name__ == "__main__":
    main()
