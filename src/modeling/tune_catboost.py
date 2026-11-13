
"""Research-only Optuna tuning for NBA DraftKings CatBoost."""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor

from src.modeling.features import CATEGORICAL_FEATURES
from src.modeling.model_adapters.catboost_adapter import (
    DEFAULT_PARAMS,
    prepare_training_features,
)
from src.modeling.splits import biweekly_walk_forward_splits


DATA_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/optuna_catboost")
STUDY_NAME = "catboost_2025_26_cpu_screen_v2"

BASELINE_PARAMS = DEFAULT_PARAMS.copy()


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.dropna(subset=["DK_POINTS"]).copy()

    X, preprocessing = prepare_training_features(df)
    categories = preprocessing["categories"]

    folds = []
    for fold_num, (train_idx, val_idx) in enumerate(
        biweekly_walk_forward_splits(df), start=1
    ):
        folds.append((
            fold_num,
            X.loc[train_idx],
            df.loc[train_idx, "DK_POINTS"],
            X.loc[val_idx],
            df.loc[val_idx, "DK_POINTS"],
        ))

    return df, X, categories, folds


def evaluate_folds(params, folds, categories):
    errors = []
    fold_results = []

    for fold_num, X_train, y_train, X_val, y_val in folds:
        print(
            f"  Training fold {fold_num} "
            f"({len(X_train):,} train, {len(X_val):,} validation)",
            flush=True,
        )

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=categories,
        )

        predictions = np.maximum(model.predict(X_val), 0)
        residuals = predictions - y_val.to_numpy()

        errors.append(residuals)

        fold_results.append({
            "FOLD": fold_num,
            "N": len(residuals),
            "MAE": np.abs(residuals).mean(),
            "RMSE": np.sqrt(np.mean(residuals ** 2)),
            "BIAS": residuals.mean(),
        })

    combined = np.concatenate(errors)

    return {
        "MAE": float(np.abs(combined).mean()),
        "RMSE": float(np.sqrt(np.mean(combined ** 2))),
        "BIAS": float(combined.mean()),
    }, fold_results


def make_objective(folds, categories):
    def objective(trial):
        params = BASELINE_PARAMS.copy()

        params.update({
            "iterations": trial.suggest_int(
                "iterations", 400, 2000, step=100
            ),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 15.0
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 3.0
            ),
        })

        metrics, fold_results = evaluate_folds(
            params, folds, categories
        )

        trial.set_user_attr("RMSE", metrics["RMSE"])
        trial.set_user_attr("BIAS", metrics["BIAS"])
        trial.set_user_attr("fold_results", fold_results)

        return metrics["MAE"]

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, X, categories, folds = load_data()

    screening_folds = [
    fold for fold in folds
    if fold[0] in (1, 3, 5)
    ]

    print(f"Loaded {len(df):,} player-games")
    print(f"Features: {X.shape[1]}")
    print(f"Categorical: {categories}")
    print(f"Walk-forward folds: {len(folds)}")

    storage = (
        f"sqlite:///{(OUTPUT_DIR / 'study.db').as_posix()}"
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Include original configuration once.
    if len(study.trials) == 0:
        study.enqueue_trial({
            "iterations": 500,
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
        })

    study.optimize(
        make_objective(folds, categories),
        n_trials=args.trials,
        gc_after_trial=True,
    )

    trials = study.trials_dataframe()
    trials.to_csv(OUTPUT_DIR / "trials.csv", index=False)

    best = study.best_trial

    print("\nBEST TRIAL")
    print(f"Trial: {best.number}")
    print(f"MAE: {best.value:.4f}")
    print(f"RMSE: {best.user_attrs['RMSE']:.4f}")
    print(f"BIAS: {best.user_attrs['BIAS']:.4f}")
    print(json.dumps(best.params, indent=2))

    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
