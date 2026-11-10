"""Research-only chronological downside-risk benchmark for frozen CatBoost predictions.

Run: python -m src.modeling.downside_risk
Does not modify production models, projections, or optimizer.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.features import PREGAME_FEATURES

PREDICTIONS_PATH = Path("reports/modeling/boosting_comparison/predictions.csv")
FEATURES_PATH = Path("data/nba_features_2025_26.csv")
OUTPUT_DIR = Path("reports/modeling/downside_risk_absolute_10")
CATEGORICAL = ["OPPONENT", "HOME_AWAY"]
KEYS = ["GAME_ID", "PLAYER_ID"]
VALIDATION_WINDOWS = [
    ("2026-02-01", "2026-02-15"),
    ("2026-02-15", "2026-03-01"),
    ("2026-03-01", "2026-03-15"),
    ("2026-03-15", "2026-04-01"),
]


def load_data():
    dtype = {key: str for key in KEYS}
    pred = pd.read_csv(PREDICTIONS_PATH, dtype=dtype)
    features = pd.read_csv(FEATURES_PATH, dtype=dtype, low_memory=False)
    required = KEYS + PREGAME_FEATURES
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if features.duplicated(KEYS).any() or pred.duplicated(KEYS).any():
        raise ValueError("Duplicate player-game keys")
    df = pred.merge(features[required], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if (df["_merge"] != "both").any():
        raise ValueError("Frozen predictions missing corresponding pregame features")
    df = df.drop(columns="_merge")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["CATBOOST"] = pd.to_numeric(df["CATBOOST"], errors="coerce")
    df["ACTUAL"] = pd.to_numeric(df["ACTUAL"], errors="coerce")
    df = df.loc[df["CATBOOST"].gt(0) & df["CATBOOST"].notna() & df["ACTUAL"].notna()].copy()
    df["DOWNSIDE"] = (
        df["ACTUAL"] <= df["CATBOOST"] - 10
    ).astype(int)
    return df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


def prepare_features(train, test):
    columns = ["CATBOOST"] + PREGAME_FEATURES
    x_train = train[columns].copy()
    x_test = test[columns].copy()
    for col in columns:
        if col in CATEGORICAL:
            train_cat = x_train[col].fillna("UNKNOWN").astype(str)
            test_cat = x_test[col].fillna("UNKNOWN").astype(str)
            categories = pd.Index(train_cat.unique())
            x_train[col] = pd.Categorical(train_cat, categories=categories)
            x_test[col] = pd.Categorical(test_cat, categories=categories)
        else:
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            x_test[col] = pd.to_numeric(x_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x_train, x_test


def score(y, p):
    return {
        "N": len(y),
        "EVENT_RATE": float(np.mean(y)),
        "AVG_PREDICTED_RISK": float(np.mean(p)),
        "BRIER": brier_score_loss(y, p),
        "LOG_LOSS": log_loss(y, p, labels=[0, 1]),
    }


def main():
    df = load_data()
    print(f"Matched player-games: {len(df):,}")
    print(f"Date range: {df.GAME_DATE.min().date()} to {df.GAME_DATE.max().date()}")
    print(f"Overall downside rate: {df.DOWNSIDE.mean():.2%}")
    predictions = []
    metrics = []
    for fold, (start, end) in enumerate(VALIDATION_WINDOWS, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        train = df.loc[df.GAME_DATE < start].copy()
        test = df.loc[(df.GAME_DATE >= start) & (df.GAME_DATE < end)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Empty training or validation period in fold {fold}")
        if train.DOWNSIDE.nunique() != 2:
            raise ValueError(f"Training labels contain only one class in fold {fold}")
        assert train.GAME_DATE.max() < test.GAME_DATE.min()
        y_train = train.DOWNSIDE.to_numpy()
        y_test = test.DOWNSIDE.to_numpy()

        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        baseline.fit(train[["CATBOOST"]], y_train)
        p_baseline = baseline.predict_proba(test[["CATBOOST"]])[:, 1]

        projection_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=100,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        projection_model.fit(train[["CATBOOST"]], y_train)

        p_projection = projection_model.predict_proba(
            test[["CATBOOST"]]
        )[:, 1]

        x_train, x_test = prepare_features(train, test)
        candidate = lgb.LGBMClassifier(
            objective="binary", n_estimators=150, learning_rate=0.03,
            num_leaves=15, max_depth=5, min_child_samples=100,
            colsample_bytree=0.85, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbosity=-1,
        )
        candidate.fit(x_train, y_train, categorical_feature=CATEGORICAL)
        p_candidate = candidate.predict_proba(x_test)[:, 1]

        result = test[["GAME_DATE", "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "CATBOOST", "ACTUAL", "DOWNSIDE"]].copy()
        result["RISK_BASELINE"] = p_baseline
        result["RISK_PROJECTION_LGBM"] = p_projection
        result["RISK_LIGHTGBM"] = p_candidate
        result["FOLD"] = fold
        predictions.append(result)
        for model, probabilities in [
            ("BASELINE", p_baseline),
            ("PROJECTION_LGBM", p_projection),
            ("LIGHTGBM", p_candidate),
        ]:
            metrics.append({"FOLD": fold, "START": start.date(), "END_EXCLUSIVE": end.date(),
                            "TRAIN_N": len(train), "MODEL": model, **score(y_test, probabilities)})
        print(f"Fold {fold}: train {len(train):,}, evaluate {len(test):,} ({start.date()} to {(end - pd.Timedelta(days=1)).date()})")

    all_predictions = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(metrics)
    overall = pd.DataFrame([
        {"MODEL": model, **score(all_predictions.DOWNSIDE, all_predictions[col])}
        for model, col in [
            ("BASELINE", "RISK_BASELINE"),
            ("PROJECTION_LGBM", "RISK_PROJECTION_LGBM"),
            ("LIGHTGBM", "RISK_LIGHTGBM"),
        ]
    ])
    calibration = []
    for model, col in [
        ("BASELINE", "RISK_BASELINE"),
        ("PROJECTION_LGBM", "RISK_PROJECTION_LGBM"),
        ("LIGHTGBM", "RISK_LIGHTGBM"),
    ]:
        tmp = all_predictions[["DOWNSIDE", col]].copy()
        tmp["RISK_QUINTILE"] = pd.qcut(tmp[col].rank(method="first"), 5, labels=False) + 1
        for quintile, group in tmp.groupby("RISK_QUINTILE"):
            calibration.append({"MODEL": model, "RISK_QUINTILE": quintile,
                                "N": len(group), "PREDICTED_RISK": group[col].mean(),
                                "ACTUAL_RATE": group.DOWNSIDE.mean()})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(OUTPUT_DIR / "calibration.csv", index=False)
    print("\nOVERALL OUT-OF-SAMPLE METRICS")
    print(overall.round(4).to_string(index=False))
    print("\nFOLD METRICS")
    print(fold_metrics[["FOLD", "MODEL", "N", "EVENT_RATE", "BRIER", "LOG_LOSS"]].round(4).to_string(index=False))
    print("\nCALIBRATION")
    print(pd.DataFrame(calibration).round(4).to_string(index=False))
    print(f"\nSaved research outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
