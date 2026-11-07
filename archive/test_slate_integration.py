from datetime import date

import pandas as pd

from src.nba_scraper import discover_scheduled_games
from src.dk_features import (
    load_player_games,
    load_team_games,
    build_features,
)
from src.data_processing.slate import (
    resolve_player_mapping,
    attach_slate_games,
)
from src.slate_features import (
    build_slate_player_rows,
    build_slate_team_rows,
)

from src.modeling.predict import predict_features

slate_date = pd.Timestamp("2026-11-07")

# Load only games played before the slate.
players = load_player_games()
teams = load_team_games()

players = players.loc[
    players["GAME_DATE"] < slate_date
].copy()

teams = teams.loc[
    teams["GAME_DATE"] < slate_date
].copy()

# Load and map the DraftKings slate.
dk = pd.read_csv(
    "data/dk_salaries/DKSalaries_2026_11_07_Main.csv"
)
directory = pd.read_csv("data/espn_player_directory.csv")
mapping = pd.read_csv("data/dk_espn_player_mapping.csv")

games = discover_scheduled_games(date(2026, 11, 7))

mapped = attach_slate_games(
    resolve_player_mapping(dk, directory, mapping),
    games,
)

# Create future placeholder records.
future_players = build_slate_player_rows(
    mapped, players.columns
)

future_teams = build_slate_team_rows(
    games, teams.columns
)

# Normalize game IDs before concatenation and merging.
players["GAME_ID"] = players["GAME_ID"].astype(str)
teams["GAME_ID"] = teams["GAME_ID"].astype(str)

future_players["GAME_ID"] = future_players["GAME_ID"].astype(str)
future_teams["GAME_ID"] = future_teams["GAME_ID"].astype(str)

# Combine history and future schedule.
all_players = pd.concat(
    [players, future_players],
    ignore_index=True,
)

all_teams = pd.concat(
    [teams, future_teams],
    ignore_index=True,
)

print("Historical player rows:", len(players))
print("Future player rows:", len(future_players))
print("Historical team rows:", len(teams))
print("Future team rows:", len(future_teams))

# Build all pregame features.
features = build_features(all_players, all_teams)

# Extract the future slate.
slate_features = features.loc[
    features["GAME_DATE"].eq(slate_date)
].copy()

# Generate production LightGBM projections.
slate_predictions = predict_features(slate_features)

print()
print("SLATE PROJECTIONS")
print(
    slate_predictions[
        ["PLAYER_NAME", "TEAM", "PROJECTED_DK_POINTS"]
    ]
    .sort_values("PROJECTED_DK_POINTS", ascending=False)
    .head(20)
    .to_string(index=False)
)

print()
print("PROJECTIONS BY PLAYER HISTORY")

print(
    slate_predictions[
        [
            "PLAYER_NAME",
            "TEAM",
            "PRIOR_GAMES",
            "DK_POINTS_AVG_SEASON",
            "MIN_AVG_SEASON",
            "PROJECTED_DK_POINTS",
        ]
    ]
    .sort_values(
        ["PRIOR_GAMES", "PROJECTED_DK_POINTS"],
        ascending=[True, False],
    )
    .head(35)
    .to_string(index=False)
)

# Check previous-season coverage for players with no current history.
previous = pd.read_csv(
    "data/nba_player_games_2025_26.csv"
)

previous["MIN"] = pd.to_numeric(
    previous["MIN"], errors="coerce"
)

previous = previous.dropna(subset=["MIN"])

previous_counts = (
    previous.groupby("PLAYER_ID")
    .size()
    .rename("PREVIOUS_SEASON_GAMES")
)

zero_history = slate_predictions.loc[
    slate_predictions["PRIOR_GAMES"].eq(0)
].copy()

zero_history = zero_history.join(
    previous_counts,
    on="PLAYER_ID",
)

zero_history["PREVIOUS_SEASON_GAMES"] = (
    zero_history["PREVIOUS_SEASON_GAMES"]
    .fillna(0)
    .astype(int)
)

print()
print("ZERO CURRENT-SEASON HISTORY")
print(
    zero_history[
        [
            "PLAYER_NAME",
            "TEAM",
            "PREVIOUS_SEASON_GAMES",
            "PROJECTED_DK_POINTS",
        ]
    ]
    .sort_values(
        "PREVIOUS_SEASON_GAMES",
        ascending=False,
    )
    .to_string(index=False)
)

print()
print("SLATE FEATURE RESULTS")
print("Rows:", len(slate_features))
print("Columns:", len(slate_features.columns))
print("Unique players:", slate_features["PLAYER_ID"].nunique())
print("Missing DK points:", slate_features["DK_POINTS"].isna().sum())

print()
print(
    slate_features[
        ["PLAYER_NAME", "TEAM", "GAME_ID"]
    ].head(10).to_string(index=False)
)

print()
print("FEATURE COMPLETENESS")

missing = (
    slate_features
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing[missing > 0]
    .to_string()
)