
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
                "iterations", 400, 900, step=100
            ),
            "depth": trial.suggest_int("depth", 5, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.015, 0.06, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 3.0, 20.0
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.5, 1.5
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

def validate_best(folds, categories, study, output_dir):
    baseline = BASELINE_PARAMS.copy()

    tuned = BASELINE_PARAMS.copy()
    tuned.update(study.best_trial.params)

    results = []

    for name, params in [
        ("baseline", baseline),
        ("tuned", tuned),
    ]:
        print(f"\nValidating {name.upper()} across all folds", flush=True)

        metrics, fold_results = evaluate_folds(
            params, folds, categories
        )

        print(
            f"{name}: MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, "
            f"BIAS={metrics['BIAS']:+.4f}"
        )

        results.append({
            "MODEL": name,
            **metrics,
        })

        pd.DataFrame(fold_results).to_csv(
            output_dir / f"validation_{name}_folds.csv",
            index=False,
        )

    pd.DataFrame(results).to_csv(
        output_dir / "validation_comparison.csv",
        index=False,
    )

    print("\nValidation comparison:")
    print(pd.DataFrame(results).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
       "--validate-best",
        action="store_true",
        help="Compare baseline and best tuned parameters across all seven folds",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA GPU for CatBoost training",
    )   
    args = parser.parse_args()

    if args.gpu:
        BASELINE_PARAMS.update({
            "task_type": "GPU",
            "devices": "0",
            "thread_count": -1,
        })
        print("CatBoost training mode: GPU")
    else:
        print("CatBoost training mode: CPU")

    output_dir = OUTPUT_DIR / (
      "gpu_focused" if args.gpu else "cpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df, X, categories, folds = load_data()

    screening_folds = [
    fold for fold in folds
    if fold[0] in (1, 3, 5)
    ]

    print(f"Loaded {len(df):,} player-games")
    print(f"Features: {X.shape[1]}")
    print(f"Categorical: {categories}")
    print(f"Walk-forward folds: {len(folds)}")

    study_name = (
        "catboost_2025_26_gpu_focused_v1"
        if args.gpu
        else STUDY_NAME
    )

    storage = (
        f"sqlite:///{(OUTPUT_DIR / 'study.db').as_posix()}"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if args.validate_best:
        validate_best(folds, categories, study, output_dir)
        return

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
        make_objective(screening_folds, categories),
        n_trials=args.trials,
        gc_after_trial=True,
    )

    trials = study.trials_dataframe()
    trials.to_csv(output_dir / "trials.csv", index=False)

    best = study.best_trial

    print("\nBEST TRIAL")
    print(f"Trial: {best.number}")
    print(f"MAE: {best.value:.4f}")
    print(f"RMSE: {best.user_attrs['RMSE']:.4f}")
    print(f"BIAS: {best.user_attrs['BIAS']:.4f}")
    print(json.dumps(best.params, indent=2))

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
