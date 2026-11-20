from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path("reports/modeling/minutes_trends")
df = pd.read_pickle(ROOT / "minutes_trend_features.pkl")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

train = df[df["GAME_DATE"] < "2026-07-01"].copy()
test = df[df["GAME_DATE"] >= "2026-10-01"].copy()

# Keep numeric pregame features only.
all_features = [
    c for c in df.columns
    if c.startswith("MIN_")
    and c != "MIN"
    and pd.api.types.is_numeric_dtype(df[c])
]

basic = [
    c for c in all_features
    if c.startswith(("MIN_AVG_", "MIN_STD_"))
]

trends = [
    c for c in all_features
    if c.startswith((
        "MIN_AVG_", "MIN_STD_",
        "MIN_DELTA_", "MIN_RATIO_",
        "MIN_SLOPE_", "MIN_LAST",
        "MIN_ROLE_SHIFT_", "MIN_WHIPSAW_",
        "MIN_POSITIVE_CHANGES_"
    ))
]

groups = {
    "BASIC": basic,
    "TRENDS": trends,
    "ALL": all_features,
}

print(f"Train: {len(train):,} player-games")
print(f"Test:  {len(test):,} player-games")
print(f"Test dates: {test.GAME_DATE.min().date()} to "
      f"{test.GAME_DATE.max().date()}")

predictions = test[[
    "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
    "TEAM", "TARGET_MINUTES", "MIN_AVG_5"
]].copy()

predictions["BASELINE"] = test["MIN_AVG_5"]

models = {}

for name, features in groups.items():
    model = LGBMRegressor(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        reg_lambda=5.0,
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["TARGET_MINUTES"]
    )

    predictions[name] = np.maximum(
        model.predict(test[features]), 0
    )
    models[name] = (model, features)

print("\n" + "=" * 65)
print("OVERALL NEXT-GAME MINUTES ACCURACY")
print("=" * 65)

for name in ["BASELINE", "BASIC", "TRENDS", "ALL"]:
    actual = predictions["TARGET_MINUTES"]
    pred = predictions[name]

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    bias = (pred - actual).mean()

    print(
        f"{name:10s} MAE={mae:6.3f} "
        f"RMSE={rmse:6.3f} BIAS={bias:+6.3f}"
    )

print("\n" + "=" * 65)
print("PLAYERS BELOW 23-MINUTE BASELINE FLOOR")
print("=" * 65)

subset = predictions[predictions["MIN_AVG_5"] < 23].copy()

print(f"Observations: {len(subset):,}")
print(
    f"Actually played 23+ minutes: "
    f"{(subset.TARGET_MINUTES >= 23).sum():,}"
)

for name in ["BASELINE", "BASIC", "TRENDS", "ALL"]:
    mae = mean_absolute_error(
        subset["TARGET_MINUTES"], subset[name]
    )

    selected = subset[subset[name] >= 23]
    hits = (
        (selected["TARGET_MINUTES"] >= 23).mean()
        if len(selected) else float("nan")
    )

    print(
        f"{name:10s} MAE={mae:6.3f} "
        f"Eligible={len(selected):4d} "
        f"23+ Hit Rate={hits:.1%}"
    )

print("\n" + "=" * 65)
print("TOP 20 FEATURES: ALL MODEL")
print("=" * 65)

model, features = models["ALL"]

importance = pd.DataFrame({
    "FEATURE": features,
    "IMPORTANCE": model.feature_importances_
}).sort_values("IMPORTANCE", ascending=False)

print(importance.head(20).to_string(index=False))

print("\n" + "=" * 65)
print("NOVEMBER 17 UTAH PLAYERS")
print("=" * 65)

sample = predictions[
    (predictions["GAME_DATE"] == "2026-11-17")
    & predictions["PLAYER_NAME"].isin([
        "Brice Sensabaugh", "Mo Bamba", "Keyonte George"
    ])
]

print(sample[[
    "PLAYER_NAME", "MIN_AVG_5",
    "BASELINE", "BASIC", "TRENDS",
    "ALL", "TARGET_MINUTES"
]].round(2).to_string(index=False))

predictions.to_csv(
    ROOT / "minutes_model_comparison.csv",
    index=False
)
importance.to_csv(
    ROOT / "feature_importance.csv",
    index=False
)

print("\nSaved evaluation results.")
