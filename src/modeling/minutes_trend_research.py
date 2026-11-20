from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("reports/modeling/minutes_trends")
OUT.mkdir(parents=True, exist_ok=True)

files = [
    "data/nba_player_games_2025_26.csv",
    "data/nba_player_games_2026_27.csv",
]

df = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in files],
    ignore_index=True
)

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
df = df.dropna(subset=["PLAYER_ID", "GAME_DATE", "MIN"])
df = df.sort_values(
    ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
).drop_duplicates(["PLAYER_ID", "GAME_ID"])

def slope(values):
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    return np.polyfit(x, y, 1)[0]

def features(group):
    group = group.copy()
    previous = group["MIN"].shift(1)
    n = previous.expanding().count()
    group["PRIOR_GAMES"] = n

    for w in [2, 3, 5, 7, 10]:
        group[f"MIN_AVG_{w}"] = previous.rolling(w).mean()
        group[f"MIN_STD_{w}"] = previous.rolling(w).std()
        group[f"MIN_SLOPE_{w}"] = previous.rolling(w).apply(
            slope, raw=True
        )

    for a, b in [
        (2, 3), (2, 5), (3, 5),
        (3, 7), (3, 10), (5, 7), (5, 10)
    ]:
        group[f"MIN_DELTA_{a}_{b}"] = (
            group[f"MIN_AVG_{a}"] - group[f"MIN_AVG_{b}"]
        )
        group[f"MIN_RATIO_{a}_{b}"] = (
            group[f"MIN_AVG_{a}"] /
            group[f"MIN_AVG_{b}"].clip(lower=1)
        )

    group["MIN_LAST"] = previous
    group["MIN_LAST_DELTA_5"] = previous - group["MIN_AVG_5"]

    group["MIN_ROLE_SHIFT_2_3"] = (
        previous.rolling(2).mean()
        - previous.shift(2).rolling(3).mean()
    )

    group["MIN_WHIPSAW_5"] = (
        group["MIN"].diff().abs().shift(1).rolling(4).mean()
    )

    group["MIN_POSITIVE_CHANGES_5"] = (
        group["MIN"].diff().gt(0).astype(float)
        .shift(1).rolling(4).sum()
    )

    # Historical ceilings, excluding the target game.
    group["MIN_CEILING_P90"] = previous.expanding(
        min_periods=10
    ).quantile(0.90)

    group["MIN_CEILING_MAX"] = previous.expanding(
        min_periods=10
    ).max().clip(upper=42)

    for ceiling in [30, 32, 34, 36, 38]:
        headroom = (ceiling - previous).clip(lower=0)
        group[f"MIN_HEADROOM_{ceiling}"] = headroom
        group[f"MIN_SAT_SLOPE_{ceiling}"] = (
            group["MIN_SLOPE_5"].clip(lower=0)
            * (headroom / ceiling)
        )

    for source in ["P90", "MAX"]:
        ceiling = group[f"MIN_CEILING_{source}"]
        headroom = (ceiling - previous).clip(lower=0)
        group[f"MIN_HEADROOM_{source}"] = headroom
        group[f"MIN_SAT_SLOPE_{source}"] = (
            group["MIN_SLOPE_5"].clip(lower=0)
            * (headroom / ceiling.clip(lower=1))
        )

    return group

result = pd.concat(
    [features(g) for _, g in df.groupby("PLAYER_ID", sort=False)],
    ignore_index=True
)

result = result[result["PRIOR_GAMES"] >= 10].copy()

result["TARGET_MINUTES"] = result["MIN"]
result["TARGET_23_PLUS"] = (
    result["TARGET_MINUTES"] >= 23
).astype(int)

result.to_pickle(OUT / "minutes_trend_features.pkl")

feature_cols = [
    c for c in result.columns
    if c.startswith("MIN_")
]

print(f"Player-games: {len(result):,}")
print(f"Candidate minutes features: {len(feature_cols)}")
print(f"Date range: {result.GAME_DATE.min()} to {result.GAME_DATE.max()}")
print(f"Players: {result.PLAYER_ID.nunique():,}")

print("\nSample November 17 players:")
sample = result[
    (result["GAME_DATE"] == "2026-11-17")
    & result["PLAYER_NAME"].isin([
        "Brice Sensabaugh", "Mo Bamba", "Keyonte George"
    ])
]

print(sample[[
    "PLAYER_NAME", "MIN_AVG_5", "MIN_AVG_3",
    "MIN_SLOPE_5", "MIN_LAST",
    "MIN_CEILING_P90", "TARGET_MINUTES"
]].to_string(index=False))

print(f"\nSaved: {OUT / 'minutes_trend_features.pkl'}")
