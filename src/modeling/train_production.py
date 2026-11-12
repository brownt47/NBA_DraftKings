
"""Train and save a production NBA DraftKings prediction model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.modeling.model_registry import (
    DEFAULT_MODEL_TYPE,
    get_model_adapter,
)


DEFAULT_DATA = [
    Path("data/nba_features_2025_26.csv"),
    Path("data/nba_features_2026_27.csv"),
]

ARTIFACT_VERSION = 2


def default_model_path(model_type):
    """Use a separate artifact file for each model."""
    return Path(f"models/{model_type}_production.joblib")


def train_production_model(
    data_path,
    model_path=None,
    cutoff_date=None,
    model_type=DEFAULT_MODEL_TYPE,
):
    """Train using completed games strictly before cutoff_date."""

    if cutoff_date is None:
        raise ValueError("A cutoff date is required.")

    model_type = model_type.lower()
    adapter = get_model_adapter(model_type)

    if model_path is None:
        model_path = default_model_path(model_type)

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

    # Prevent training on duplicate player-games.
    if {"GAME_ID", "PLAYER_ID"}.issubset(df.columns):
        duplicates = df.duplicated(
            subset=["GAME_ID", "PLAYER_ID"]
        )

        if duplicates.any():
            raise ValueError(
                f"Found {duplicates.sum()} duplicate player-game rows."
            )

    X, preprocessing = adapter.prepare_training_features(df)
    y = pd.to_numeric(df["DK_POINTS"], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    model = adapter.create_model()

    if model_type == "catboost":
        model.fit(
            X,
            y,
            cat_features=preprocessing["categories"],
        )
    else:
        model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": model_type,
        "model": model,
        "features": list(X.columns),
        "preprocessing": preprocessing,
        "cutoff_date": str(cutoff.date()),
        "training_rows": len(X),
    }

    # Preserve compatibility with the existing LightGBM
    # artifact metadata convention.
    if model_type == "lightgbm":
        artifact["categories"] = preprocessing["categories"]

    joblib.dump(artifact, model_path)

    print(f"Model type: {model_type}")
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

    parser.add_argument(
        "--model-type",
        choices=["catboost", "lightgbm"],
        default=DEFAULT_MODEL_TYPE,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional output path for the trained model.",
    )

    parser.add_argument("--cutoff", required=True)

    args = parser.parse_args()

    train_production_model(
        data_path=args.data,
        model_path=args.model,
        cutoff_date=args.cutoff,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
