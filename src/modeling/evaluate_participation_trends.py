from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("reports/modeling/minutes_trends")
OUT.mkdir(parents=True, exist_ok=True)

files = [
    ("2025_26", "data/nba_player_participation_2025_26.csv"),
    ("2026_27", "data/nba_player_participation_2026_27.csv"),
]

frames = []
for season, path in files:
    d = pd.read_csv(path, low_memory=False)
    d["SEASON"] = season
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

df["PLAYED"] = (
    df["PLAYED"].astype(str).str.lower().eq("true")
)
df["DID_NOT_PLAY"] = (
    df["DID_NOT_PLAY"].astype(str).str.lower().eq("true")
)

df["MIN_NUM"] = pd.to_numeric(df["MIN"], errors="coerce")
df["MIN_TEAM"] = np.where(df["PLAYED"], df["MIN_NUM"], 0.0)

df = df.sort_values([
    "SEASON", "PLAYER_ID", "TEAM", "GAME_DATE", "GAME_ID"
])

def slope(x):
    return np.polyfit(np.arange(len(x)), x, 1)[0]

def make_features(g):
    g = g.copy()
    mins = g["MIN_TEAM"]
    played = g["PLAYED"].astype(float)

    # Shift before rolling: current game is never an input.
    prior = mins.shift(1)
    prior_played = played.shift(1)

    g["PRIOR_LISTINGS"] = np.arange(len(g))
    g["MIN_AVG_TEAM_5"] = prior.rolling(5).mean()
    g["MIN_SLOPE_TEAM_5"] = prior.rolling(5).apply(
        slope, raw=True
    )
    g["APPEAR_RATE_5"] = prior_played.rolling(5).mean()
    g["APPEAR_RATE_10"] = prior_played.rolling(10).mean()
    g["DNP_RATE_5"] = 1 - g["APPEAR_RATE_5"]

    # Last five played appearances, excluding current game.
    played_minutes = mins.where(g["PLAYED"])
    played_history = played_minutes.dropna()

    slope_history = played_history.rolling(5).apply(
        slope, raw=True
    )

    # Map each completed played-game slope to its record,
    # then forward-fill and shift to exclude the current game.
    g["MIN_SLOPE_PLAYED_5"] = (
        slope_history.reindex(g.index).ffill().shift(1)
    )

    g["TARGET_PLAYED"] = g["PLAYED"].astype(int)
    g["TARGET_MINUTES"] = mins
    g["TARGET_23_PLUS"] = (mins >= 23).astype(int)

    return g

result = pd.concat([
    make_features(g)
    for _, g in df.groupby(
        ["SEASON", "PLAYER_ID", "TEAM"], sort=False
    )
], ignore_index=True)

# Require at least five prior recorded listings and
# a valid five-played-game slope.
result = result[
    (result["PRIOR_LISTINGS"] >= 5)
    & result["MIN_SLOPE_PLAYED_5"].notna()
].copy()

bins = [-np.inf, -3, -1, 1, 3, np.inf]
labels = [
    "Strong decline",
    "Moderate decline",
    "Stable",
    "Moderate rise",
    "Strong rise",
]

result["TREND_GROUP"] = pd.cut(
    result["MIN_SLOPE_PLAYED_5"],
    bins=bins,
    labels=labels,
    right=False
)

def summarize(d):
    return pd.Series({
        "N": len(d),
        "DNP_RATE": 1 - d["TARGET_PLAYED"].mean(),
        "REACHED_23_RATE": d["TARGET_23_PLUS"].mean(),
        "AVG_NEXT_MIN": d["TARGET_MINUTES"].mean(),
        "AVG_PRIOR_DNP_RATE": d["DNP_RATE_5"].mean(),
    })

print("=" * 75)
print("NEXT RECORDED TEAM-GAME OUTCOMES BY PREGAME MINUTES SLOPE")
print("=" * 75)

summary = result.groupby(
    "TREND_GROUP", observed=True
).apply(summarize, include_groups=False)

print(summary.round(3).to_string())

print("\n" + "=" * 75)
print("RISING PLAYERS BY RECENT PARTICIPATION")
print("=" * 75)

rising = result[
    result["MIN_SLOPE_PLAYED_5"] >= 1
].copy()

rising["PARTICIPATION_GROUP"] = pd.cut(
    rising["APPEAR_RATE_5"],
    bins=[-0.01, 0.6, 0.8, 0.999, 1.01],
    labels=[
        "60% or less",
        "61-80%",
        "81-99%",
        "100%"
    ]
)

print(
    rising.groupby(
        "PARTICIPATION_GROUP", observed=True
    ).apply(summarize, include_groups=False)
    .round(3).to_string()
)

print("\n" + "=" * 75)
print("2026-27 ONLY")
print("=" * 75)

current = result[result["SEASON"] == "2026_27"]

print(
    current.groupby(
        "TREND_GROUP", observed=True
    ).apply(summarize, include_groups=False)
    .round(3).to_string()
)

result.to_pickle(OUT / "participation_trend_features.pkl")
summary.to_csv(OUT / "participation_trend_summary.csv")

print(f"\nSaved {len(result):,} observations.")
