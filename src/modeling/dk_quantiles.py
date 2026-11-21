"""Walk-forward LightGBM DK-point quantile regression."""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss

from src.modeling.lightgbm_model import _prepare_fold_features
from src.modeling.splits import biweekly_walk_forward_splits

DATA = Path("data/nba_features_2025_26.csv")
CAT = Path("reports/modeling/boosting_comparison/predictions.csv")
OUT = Path("reports/modeling/dk_quantiles")
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.10, 0.20]
KEYS = ["GAME_ID", "PLAYER_ID"]

df = pd.read_csv(
    DATA, dtype={k: str for k in KEYS}
)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

cat = pd.read_csv(
    CAT, dtype={k: str for k in KEYS}
)

cat["GAME_DATE"] = pd.to_datetime(cat["GAME_DATE"])

assert not cat.duplicated(KEYS).any()

results = []

for fold, (train_idx, val_idx) in enumerate(
    biweekly_walk_forward_splits(df), start=1
):
    train = df.loc[train_idx].dropna(
        subset=["DK_POINTS"]
    ).copy()

    val = df.loc[val_idx].dropna(
        subset=["DK_POINTS"]
    ).copy()

    assert train["GAME_DATE"].max() < val["GAME_DATE"].min()

    X_train, X_val = _prepare_fold_features(train, val)

    print(
        f"Fold {fold}: "
        f"train={len(train):,}, val={len(val):,}",
        flush=True
    )

    result = val[
        ["GAME_DATE", "GAME_ID", "PLAYER_ID", "DK_POINTS"]
    ].copy()

    result = result.rename(
        columns={"DK_POINTS": "ACTUAL"}
    )

    result["FOLD"] = fold

    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=50,
            reg_lambda=5.0,
            verbosity=-1,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, train["DK_POINTS"])

        result[f"Q{int(q*100)}"] = np.maximum(
            model.predict(X_val), 0
        )

    results.append(result)

pred = pd.concat(results, ignore_index=True)

pred = pred.merge(
    cat[KEYS + ["FOLD", "CATBOOST"]],
    on=KEYS + ["FOLD"],
    validate="one_to_one"
)

# Baselines use only earlier OOF CatBoost residuals.
pred["BASE_Q10"] = np.nan
pred["BASE_Q20"] = np.nan

for fold in sorted(pred["FOLD"].unique())[1:]:
    previous = pred[pred["FOLD"] < fold]
    current = pred["FOLD"] == fold

    residual = previous["ACTUAL"] - previous["CATBOOST"]

    for q in QUANTILES:
        offset = residual.quantile(q)

        pred.loc[current, f"BASE_Q{int(q*100)}"] = (
            pred.loc[current, "CATBOOST"] + offset
        ).clip(lower=0)

pred.to_csv(OUT / "predictions.csv", index=False)

print("\nQUANTILE RESULTS: FOLDS 2-7")

evaluation = pred[pred["FOLD"] >= 2].copy()
summary = []

for q in QUANTILES:
    for name in [f"BASE_Q{int(q*100)}", f"Q{int(q*100)}"]:
        y = evaluation["ACTUAL"]
        estimate = evaluation[name]

        summary.append({
            "QUANTILE": q,
            "MODEL": name,
            "N": len(evaluation),
            "PINBALL": mean_pinball_loss(
                y, estimate, alpha=q
            ),
            "COVERAGE": 100 * (y <= estimate).mean(),
            "AVG_FLOOR": estimate.mean(),
            "AVG_GAP": (
                evaluation["CATBOOST"] - estimate
            ).mean()
        })

s = pd.DataFrame(summary)
s.to_csv(OUT / "summary.csv", index=False)

print(s.round(3).to_string(index=False))
print(f"\nSaved to {OUT}")
