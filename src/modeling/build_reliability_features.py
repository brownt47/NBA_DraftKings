from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("reports/modeling/reliability")
OUT.mkdir(parents=True, exist_ok=True)

sources = {
    "2025_26": "data/nba_player_participation_2025_26.csv",
    "2026_27": "data/nba_player_participation_2026_27.csv",
}

frames = []

for season, path in sources.items():
    d = pd.read_csv(path, low_memory=False)
    d["SEASON"] = season
    frames.append(d)

df = pd.concat(frames, ignore_index=True)

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

for col in ["PLAYED", "DID_NOT_PLAY", "STARTER"]:
    df[col] = (
        df[col].astype(str).str.lower()
        .map({"true": True, "false": False})
    )

df["MIN_NUM"] = pd.to_numeric(df["MIN"], errors="coerce")

df["MIN_TEAM"] = np.where(
    df["PLAYED"], df["MIN_NUM"], 0.0
)

df["COACH_DNP"] = (
    df["DID_NOT_PLAY"]
    & df["DNP_REASON"].fillna("")
        .str.contains("COACH", case=False, regex=False)
)

df = df.sort_values([
    "SEASON", "PLAYER_ID", "TEAM",
    "GAME_DATE", "GAME_ID"
]).reset_index(drop=True)

def slope(values):
    y = np.asarray(values, dtype=float)
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def streak(values, target=1):
    count = 0
    for v in values[::-1]:
        if v == target:
            count += 1
        else:
            break
    return count

def build_features(g):
    g = g.copy()

    played = g["PLAYED"].astype(float)
    mins = g["MIN_TEAM"]
    started = g["STARTER"].astype(float)
    coach = g["COACH_DNP"].astype(float)

    prior_played = played.shift(1)
    prior_mins = mins.shift(1)
    prior_started = started.shift(1)
    prior_coach = coach.shift(1)

    g["PRIOR_LISTINGS"] = np.arange(len(g))

    for w in [3, 5, 10]:
        g[f"APPEAR_RATE_{w}"] = (
            prior_played.rolling(w).mean()
        )

        g[f"DNP_RATE_{w}"] = (
            1 - g[f"APPEAR_RATE_{w}"]
        )

        g[f"COACH_DNP_RATE_{w}"] = (
            prior_coach.rolling(w).mean()
        )

        g[f"START_RATE_{w}"] = (
            prior_started.rolling(w).mean()
        )

        g[f"MIN_TEAM_AVG_{w}"] = (
            prior_mins.rolling(w).mean()
        )

        g[f"MIN_TEAM_STD_{w}"] = (
            prior_mins.rolling(w).std()
        )

        g[f"MIN_TEAM_SLOPE_{w}"] = (
            prior_mins.rolling(w).apply(
                slope, raw=True
            )
        )

        for threshold in [15, 20, 23, 25, 30]:
            g[f"MIN_{threshold}_PLUS_RATE_{w}"] = (
                prior_mins.ge(threshold)
                .astype(float)
                .rolling(w).mean()
            )

    g["LAST_PLAYED"] = prior_played
    g["LAST_STARTED"] = prior_started
    g["LAST_MINUTES"] = prior_mins

    g["APPEAR_STREAK"] = (
        prior_played.expanding().apply(
            lambda x: streak(x, 1), raw=True
        )
    )

    g["DNP_STREAK"] = (
        prior_played.expanding().apply(
            lambda x: streak(x, 0), raw=True
        )
    )

    # Played-game-only history, excluding current game.
    played_minutes = mins.where(g["PLAYED"])

    for w in [3, 5, 10]:
        history = played_minutes.dropna()

        rolling_avg = history.rolling(w).mean()
        rolling_std = history.rolling(w).std()
        rolling_slope = history.rolling(w).apply(
            slope, raw=True
        )

        g[f"MIN_PLAYED_AVG_{w}"] = (
            rolling_avg.reindex(g.index).ffill().shift(1)
        )

        g[f"MIN_PLAYED_STD_{w}"] = (
            rolling_std.reindex(g.index).ffill().shift(1)
        )

        g[f"MIN_PLAYED_SLOPE_{w}"] = (
            rolling_slope.reindex(g.index).ffill().shift(1)
        )

    g["MIN_DELTA_PLAYED_3_5"] = (
        g["MIN_PLAYED_AVG_3"]
        - g["MIN_PLAYED_AVG_5"]
    )

    g["MIN_DELTA_TEAM_3_5"] = (
        g["MIN_TEAM_AVG_3"]
        - g["MIN_TEAM_AVG_5"]
    )

    g["TARGET_PLAYED"] = g["PLAYED"].astype(int)
    g["TARGET_MINUTES"] = mins
    g["TARGET_23_PLUS"] = (
        mins >= 23
    ).astype(int)

    g["TARGET_25_PLUS"] = (
        mins >= 25
    ).astype(int)

    return g

result = pd.concat(
    [
        build_features(g)
        for _, g in df.groupby(
            ["SEASON", "PLAYER_ID", "TEAM"],
            sort=False
        )
    ],
    ignore_index=True
)

# At least 10 previous recorded appearances/DNP listings.
result = result[
    result["PRIOR_LISTINGS"] >= 10
].copy()

result.to_pickle(
    OUT / "player_reliability_features.pkl"
)

print("=" * 70)
print("RELIABILITY FEATURE DATASET")
print("=" * 70)

print(f"Observations: {len(result):,}")
print(f"Players: {result.PLAYER_ID.nunique():,}")
print(f"Seasons: {result.SEASON.nunique()}")
print(f"Recorded DNPs: {(result.TARGET_PLAYED == 0).sum():,}")

feature_cols = [
    c for c in result.columns
    if c.startswith((
        "APPEAR_", "DNP_", "COACH_DNP_RATE_",
        "START_RATE_", "LAST_", "MIN_"
    ))
    and pd.api.types.is_numeric_dtype(result[c])
]

print(f"Candidate numeric features: {len(feature_cols)}")

print("\n2026-27 November 17 Utah players:")

sample = result[
    (result["GAME_DATE"] == "2026-11-17")
    & result["PLAYER_NAME"].isin([
        "Brice Sensabaugh",
        "Mo Bamba",
        "Keyonte George"
    ])
]

cols = [
    "PLAYER_NAME",
    "APPEAR_RATE_5",
    "COACH_DNP_RATE_5",
    "START_RATE_5",
    "APPEAR_STREAK",
    "MIN_PLAYED_AVG_5",
    "MIN_PLAYED_SLOPE_5",
    "TARGET_PLAYED",
    "TARGET_MINUTES",
]

print(sample[cols].round(2).to_string(index=False))

print("\nSaved:")
print(OUT / "player_reliability_features.pkl")
