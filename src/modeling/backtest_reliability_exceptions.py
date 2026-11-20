"""Backtest reliability-based exceptions to the minutes floor."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("reports/modeling/reliability")
pred = pd.read_csv(ROOT / "reliability_v1_predictions.csv")
features = pd.read_pickle(ROOT / "player_reliability_features.pkl")

pred["GAME_DATE"] = pd.to_datetime(pred["GAME_DATE"])
features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])

keys = ["GAME_DATE", "PLAYER_ID", "TEAM"]

actual = features[
    keys + ["TARGET_MINUTES"]
].copy()

pred = pred.merge(
    actual,
    on=keys,
    how="left",
    validate="one_to_one",
)

below = pred[
    pred.MIN_PLAYED_AVG_5.notna()
    & (pred.MIN_PLAYED_AVG_5 < 23)
    & pred.TARGET_MINUTES.notna()
].copy()

print("=" * 85)
print("RELIABILITY EXCEPTION BACKTEST")
print("=" * 85)
print(f"Below-floor observations: {len(below):,}")
print(f"Dates: {below.GAME_DATE.nunique()}")
print(f"Baseline actual 23+ rate: {(below.TARGET_MINUTES >= 23).mean():.1%}")
print(f"Baseline actual below-15 rate: {(below.TARGET_MINUTES < 15).mean():.1%}")

rows = []

for model in ["BASELINE", "PARTICIPATION", "FULL"]:
    col = f"MIN23_{model}"

    for threshold in np.arange(0.40, 0.701, 0.05):
        selected = below[below[col] >= threshold]

        if selected.empty:
            continue

        rows.append({
            "MODEL": model,
            "THRESHOLD": round(float(threshold), 2),
            "N": len(selected),
            "DATES": selected.GAME_DATE.nunique(),
            "AVG_PER_DATE": len(selected) / below.GAME_DATE.nunique(),
            "HIT_23": (selected.TARGET_MINUTES >= 23).mean(),
            "BELOW_20": (selected.TARGET_MINUTES < 20).mean(),
            "BELOW_15": (selected.TARGET_MINUTES < 15).mean(),
            "AVG_MINUTES": selected.TARGET_MINUTES.mean(),
        })

results = pd.DataFrame(rows)

print("\nOVERALL THRESHOLD COMPARISON")
print(
    results.to_string(
        index=False,
        formatters={
            "HIT_23": "{:.1%}".format,
            "BELOW_20": "{:.1%}".format,
            "BELOW_15": "{:.1%}".format,
            "AVG_PER_DATE": "{:.2f}".format,
            "AVG_MINUTES": "{:.1f}".format,
        },
    )
)

print("\n" + "=" * 85)
print("FULL MODEL AT 50%: DATE-LEVEL RESULTS")
print("=" * 85)

selected = below[below.MIN23_FULL >= 0.50].copy()

daily = selected.groupby("GAME_DATE").agg(
    N=("PLAYER_ID", "size"),
    HIT_23=("TARGET_MINUTES", lambda x: (x >= 23).mean()),
    BELOW_15=("TARGET_MINUTES", lambda x: (x < 15).mean()),
    AVG_MINUTES=("TARGET_MINUTES", "mean"),
).reset_index()

print(
    daily.to_string(
        index=False,
        formatters={
            "HIT_23": "{:.0%}".format,
            "BELOW_15": "{:.0%}".format,
            "AVG_MINUTES": "{:.1f}".format,
        },
    )
)

print("\n" + "=" * 85)
print("FULL MODEL AT 50%: INDIVIDUAL EXCEPTIONS")
print("=" * 85)

print(
    selected[
        [
            "GAME_DATE",
            "PLAYER_NAME",
            "TEAM",
            "MIN_PLAYED_AVG_5",
            "MIN_PLAYED_SLOPE_5",
            "MIN23_FULL",
            "TARGET_MINUTES",
        ]
    ]
    .sort_values(["GAME_DATE", "MIN23_FULL"])
    .round(2)
    .to_string(index=False)
)

results.to_csv(ROOT / "reliability_exception_backtest.csv", index=False)
daily.to_csv(ROOT / "reliability_exception_daily.csv", index=False)

print("\nSaved backtest results.")
