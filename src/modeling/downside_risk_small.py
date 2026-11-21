"""Walk-forward LightGBM downside-risk classification."""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
from src.modeling.features import prepare_model_features

DATA = Path("data/nba_features_2025_26.csv")
PRED = Path("reports/modeling/boosting_comparison/predictions.csv")
OUT = Path("reports/modeling/downside_risk_small")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["GAME_ID", "PLAYER_ID"]
TARGETS = ["ABS_10", "REL_30", "REL_50"]

p = pd.read_csv(PRED, dtype={k: str for k in KEYS})
f = pd.read_csv(DATA, dtype={k: str for k in KEYS})

d = p.merge(
    f.drop(columns=["GAME_DATE", "PLAYER_NAME"], errors="ignore"),
    on=KEYS,
    validate="one_to_one",
    suffixes=("", "_FEATURE")
)

d = d.dropna(subset=["CATBOOST", "ACTUAL"])
d = d[d["CATBOOST"] >= 10].copy()

miss = d["CATBOOST"] - d["ACTUAL"]
d["ABS_10"] = (miss >= 10).astype(int)
d["REL_30"] = (miss / d["CATBOOST"] >= .30).astype(int)
d["REL_50"] = (miss / d["CATBOOST"] >= .50).astype(int)

X = d.copy()

risk_features = [
    "MIN_STD_LAST_5",
    "MIN_STD_LAST_10",
    "MIN_WHIPSAW_5",
    "DK_WHIPSAW_5",
    "PARTICIPATION_COVERAGE_LAST_5",
    "PARTICIPATION_COVERAGE_LAST_10",
    "TEAM_ROTATION_CONCENTRATION_AVG_5",
]

missing = [c for c in risk_features if c not in X.columns]
if missing:
    raise ValueError(f"Missing risk features: {missing}")

X = X[risk_features].copy()
X["CATBOOST_PROJECTION"] = d["CATBOOST"].to_numpy()

for col in X.select_dtypes(include=["category"]).columns:
    X[col] = X[col].cat.add_categories(
        ["UNKNOWN"]
    ).fillna("UNKNOWN")

X = X.replace([np.inf, -np.inf], np.nan)

print(f"Eligible games: {len(d):,}")
print(f"Predictors: {X.shape[1]}")
print(f"Folds: {sorted(d.FOLD.unique())}")

results = []
predictions = []

params = dict(
    n_estimators=250,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=60,
    reg_lambda=5.0,
    verbosity=-1,
    random_state=42,
    n_jobs=-1
)

for fold in sorted(d.FOLD.unique())[1:]:
    train = d["FOLD"] < fold
    test = d["FOLD"] == fold

    assert d.loc[train, "GAME_DATE"].max() < \
           d.loc[test, "GAME_DATE"].min()

    print(
        f"\nFold {fold}: "
        f"train={train.sum():,}, test={test.sum():,}",
        flush=True
    )

    for target in TARGETS:
        y_train = d.loc[train, target]
        y_test = d.loc[test, target]

        for model_name, columns in [
            ("PROJECTION_ONLY", ["CATBOOST_PROJECTION"]),
            ("FULL_FEATURES", list(X.columns))
        ]:
            model = lgb.LGBMClassifier(**params)

            model.fit(
                X.loc[train, columns],
                y_train
            )

            prob = model.predict_proba(
                X.loc[test, columns]
            )[:, 1]

            prob = np.clip(prob, 1e-6, 1 - 1e-6)

            results.append({
                "FOLD": fold,
                "TARGET": target,
                "MODEL": model_name,
                "N": int(test.sum()),
                "EVENTS": int(y_test.sum()),
                "ROC_AUC": roc_auc_score(y_test, prob),
                "PR_AUC": average_precision_score(y_test, prob),
                "BRIER": brier_score_loss(y_test, prob),
                "LOG_LOSS": log_loss(y_test, prob)
            })

            predictions.append(pd.DataFrame({
                "GAME_ID": d.loc[test, "GAME_ID"].to_numpy(),
                "PLAYER_ID": d.loc[test, "PLAYER_ID"].to_numpy(),
                "GAME_DATE": d.loc[test, "GAME_DATE"].to_numpy(),
                "FOLD": fold,
                "TARGET": target,
                "MODEL": model_name,
                "ACTUAL_EVENT": y_test.to_numpy(),
                "PROBABILITY": prob
            }))

r = pd.DataFrame(results)
o = pd.concat(predictions, ignore_index=True)

r.to_csv(OUT / "fold_metrics.csv", index=False)
o.to_csv(OUT / "predictions.csv", index=False)

print("\nPOOLED OUT-OF-SAMPLE RESULTS")

summary = []

for (target, name), g in o.groupby(["TARGET", "MODEL"]):
    y = g["ACTUAL_EVENT"]
    prob = g["PROBABILITY"]

    summary.append({
        "TARGET": target,
        "MODEL": name,
        "N": len(g),
        "EVENT_RATE": 100 * y.mean(),
        "ROC_AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "BRIER": brier_score_loss(y, prob),
        "LOG_LOSS": log_loss(y, prob)
    })

s = pd.DataFrame(summary)
s.to_csv(OUT / "summary.csv", index=False)

print(s.round(4).to_string(index=False))
print(f"\nSaved results to {OUT}")
