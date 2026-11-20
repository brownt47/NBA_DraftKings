"""Evaluate NBA participation and minutes reliability models."""

from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
)

DATA = Path("reports/modeling/reliability/player_reliability_features.pkl")
OUT = Path("reports/modeling/reliability")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(DATA).sort_values("GAME_DATE").copy()

train = df[df.SEASON == "2025_26"].copy()
test = df[df.SEASON == "2026_27"].copy()

baseline = [
    "MIN_PLAYED_AVG_3",
    "MIN_PLAYED_AVG_5",
    "MIN_PLAYED_AVG_10",
    "MIN_PLAYED_STD_3",
    "MIN_PLAYED_STD_5",
    "MIN_PLAYED_STD_10",
]

participation = baseline + [
    "PRIOR_LISTINGS",
    "APPEAR_RATE_3",
    "APPEAR_RATE_5",
    "APPEAR_RATE_10",
    "COACH_DNP_RATE_3",
    "COACH_DNP_RATE_5",
    "COACH_DNP_RATE_10",
    "START_RATE_3",
    "START_RATE_5",
    "START_RATE_10",
    "APPEAR_STREAK",
    "DNP_STREAK",
    "LAST_PLAYED",
    "LAST_STARTED",
]

excluded = {
    "MIN_NUM",
    "MIN_TEAM",
    "TARGET_MINUTES",
    "TARGET_PLAYED",
    "TARGET_23_PLUS",
    "TARGET_25_PLUS",
}

full = [
    c for c in df.columns
    if (
        c.startswith((
            "MIN_", "APPEAR_", "DNP_",
            "COACH_DNP_RATE_", "START_RATE_",
            "LAST_", "PRIOR_LISTINGS"
        ))
        and c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    )
]

feature_sets = {
    "BASELINE": baseline,
    "PARTICIPATION": participation,
    "FULL": full,
}

targets = {
    "PLAYED": "TARGET_PLAYED",
    "MIN23": "TARGET_23_PLUS",
}

results = []
predictions = test[[
    "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
    "TEAM", "TARGET_PLAYED", "TARGET_23_PLUS",
    "MIN_PLAYED_AVG_5", "APPEAR_RATE_5",
    "MIN_PLAYED_SLOPE_5",
]].copy()

print("=" * 75)
print("RELIABILITY MODEL V1")
print("=" * 75)
print(f"Train: {len(train):,} player-games")
print(f"Test:  {len(test):,} player-games")
print(f"Features: {len(full)} in FULL model")

for target_name, target_col in targets.items():
    print(f"\n{'=' * 75}")
    print(f"TARGET: {target_name}")
    print("=" * 75)

    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate:  {y_test.mean():.3f}")

    for model_name, features in feature_sets.items():
        model = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=5,
            min_child_samples=75,
            reg_lambda=5.0,
            verbosity=-1,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(train[features], y_train)

        p = model.predict_proba(test[features])[:, 1]
        predictions[f"{target_name}_{model_name}"] = p

        brier = brier_score_loss(y_test, p)
        auc = (
            roc_auc_score(y_test, p)
            if y_test.nunique() == 2 else np.nan
        )
        ll = log_loss(y_test, p, labels=[0, 1])

        results.append({
            "TARGET": target_name,
            "MODEL": model_name,
            "N": len(test),
            "BRIER": brier,
            "AUC": auc,
            "LOGLOSS": ll,
        })

        print(
            f"{model_name:15s} "
            f"Brier={brier:.4f} "
            f"AUC={auc:.4f} "
            f"LogLoss={ll:.4f}"
        )

metrics = pd.DataFrame(results)
metrics.to_csv(OUT / "reliability_v1_metrics.csv", index=False)
predictions.to_csv(OUT / "reliability_v1_predictions.csv", index=False)

print("\n" + "=" * 75)
print("BELOW 23-MINUTE HISTORICAL FLOOR")
print("=" * 75)

below = predictions[
    predictions["MIN_PLAYED_AVG_5"].notna()
    & (predictions["MIN_PLAYED_AVG_5"] < 23)
].copy()

print(f"Observations: {len(below):,}")
print(f"Actually reached 23: {below.TARGET_23_PLUS.mean():.1%}")

for model_name in feature_sets:
    col = f"MIN23_{model_name}"

    print(f"\n{model_name}")

    for threshold in [0.50, 0.60, 0.70, 0.80]:
        selected = below[col] >= threshold

        if selected.sum() == 0:
            continue

        precision = precision_score(
            below.TARGET_23_PLUS,
            selected,
            zero_division=0,
        )

        recall = recall_score(
            below.TARGET_23_PLUS,
            selected,
            zero_division=0,
        )

        print(
            f"  P>= {threshold:.0%}: "
            f"Selected={selected.sum():4d} "
            f"Hit23={precision:.1%} "
            f"Recall={recall:.1%}"
        )

print("\n" + "=" * 75)
print("NOVEMBER 17 UTAH CASE STUDY")
print("=" * 75)

sample = predictions[
    (predictions.GAME_DATE == "2026-11-17")
    & predictions.PLAYER_NAME.isin([
        "Brice Sensabaugh",
        "Mo Bamba",
        "Keyonte George",
    ])
]

cols = [
    "PLAYER_NAME",
    "MIN_PLAYED_AVG_5",
    "APPEAR_RATE_5",
    "MIN_PLAYED_SLOPE_5",
    "MIN23_BASELINE",
    "MIN23_PARTICIPATION",
    "MIN23_FULL",
    "TARGET_23_PLUS",
]

print(sample[cols].round(3).to_string(index=False))

print("\nSaved:")
print(OUT / "reliability_v1_metrics.csv")
print(OUT / "reliability_v1_predictions.csv")
