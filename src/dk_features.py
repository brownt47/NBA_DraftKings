
"""Feature engineering for NBA DraftKings predictions."""

from pathlib import Path
import argparse
import pandas as pd

from src.features.scoring import add_historical_scoring_features
from src.features.minutes import add_historical_minutes_features
from src.features.momentum import add_historical_momentum_features
from src.features.team import add_historical_team_features
from src.features.matchup import add_historical_matchup_features
from src.features.team_change import add_historical_team_change_features
from src.features.schedule import (
    add_historical_schedule_features,
    add_player_workload_features,
    add_player_recent_minutes,
    add_historical_travel_features,
    add_opponent_schedule_features,
)
from src.features.rotation import (
    add_historical_rotation_features,
    add_historical_player_minutes_share,
)
from src.features.usage import add_historical_usage_features
from src.features.config import ROLLING_WINDOWS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PLAYER_FILE = DATA_DIR / "nba_player_games_2026_27.csv"
TEAM_FILE = DATA_DIR / "nba_team_games_2026_27.csv"



def load_player_games(path=PLAYER_FILE):
    """Load actual player appearances in chronological order."""
    df = pd.read_csv(path)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    if df.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate player-game records detected.")

    # Exclude records with missing or nonnumeric playing time.
    minutes = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.loc[minutes.notna()].copy()

    return (
        df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )



def load_team_games(path=TEAM_FILE):
    """Load team boxscores in chronological order."""
    df = pd.read_csv(path)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    if df.duplicated(["GAME_ID", "TEAM"]).any():
        raise ValueError("Duplicate team-game records detected.")

    return (
        df.sort_values(["TEAM", "GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )



def build_features(players, teams, windows=ROLLING_WINDOWS):
    """Build pregame features from supplied player and team data."""

    players = players.copy()
    teams = teams.copy()

    # Player history
    players = add_historical_scoring_features(
        players, windows=windows
    )

    # Offensive usage and fantasy efficiency
    players = add_historical_usage_features(players)

    players = add_historical_minutes_features(
        players, windows=windows
    )

    players = add_historical_team_change_features(players)
    
    # Rotation and playing-time distribution
    players = add_historical_rotation_features(players, teams)
    players = add_historical_player_minutes_share(players, teams)

    teams = add_historical_schedule_features(teams)
    teams = add_historical_travel_features(teams)

    players = add_historical_momentum_features(players)

    # Team history
    teams = add_historical_team_features(
        teams, windows=windows
    )

    # Matchup history
    players = add_historical_matchup_features(players, teams)

    # Attach schedule features using the player's team and game.
    schedule_columns = [
        "GAME_ID",
        "TEAM",
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "GAMES_LAST_7_DAYS",
        "GAMES_LAST_14_DAYS",
        "ROAD_TRIP_GAME_NUMBER",
        "IS_FIRST_HOME_AFTER_TRIP",
        "TRAVEL_MILES",
        "TIMEZONE_CHANGE",
        "TIMEZONES_CROSSED",
        "TRAVEL_MILES_LAST_7_DAYS",
        "TRAVEL_MILES_LAST_14_DAYS",
        "TRAVEL_MILES_PER_RECOVERY_DAY",
    ]

    players = players.merge(
        teams[schedule_columns],
        on=["GAME_ID", "TEAM"],
        how="left",
        validate="many_to_one",
    )

    # Opponent rest and travel conditions
    players = add_opponent_schedule_features(players, teams)

    players = add_player_workload_features(players)

    players = add_player_recent_minutes(players)

    return players



def build_historical_features(
    windows=ROLLING_WINDOWS,
    season="2026-27",
):
    """Build the historical NBA player-game feature dataset."""

    if season not in ("2025-26", "2026-27"):
        raise ValueError(f"Unsupported season: {season}")

    player_file = DATA_DIR / f"nba_player_games_{season.replace('-', '_')}.csv"
    team_file = DATA_DIR / f"nba_team_games_{season.replace('-', '_')}.csv"

    players = load_player_games(player_file)
    teams = load_team_games(team_file)

    # Exclude exhibition games absent from the official team dataset.
    if season == "2025-26":
        valid_game_ids = set(teams["GAME_ID"].astype(str))
        players = players.loc[
            players["GAME_ID"].astype(str).isin(valid_game_ids)
        ].copy()

    return build_features(players, teams, windows=windows)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build NBA DraftKings historical features."
    )
    parser.add_argument(
        "--season",
        choices=["2025-26", "2026-27"],
        default="2026-27",
    )
    args = parser.parse_args()

    features = build_historical_features(season=args.season)

    print(f"Player-game records: {len(features):,}")
    print(f"Total columns: {len(features.columns)}")

    output = DATA_DIR / f"nba_features_{args.season.replace('-', '_')}.csv"
    features.to_csv(output, index=False)

    print(f"Saved: {output}")
