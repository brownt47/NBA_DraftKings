
from pathlib import Path
import pandas as pd
import argparse
import re

from features.scoring import (
    calculate_scoring_features,
    calculate_rolling_scoring_features,
)
from features.minutes import calculate_minutes_features
from features.momentum import calculate_momentum_features
from features.team import calculate_team_features
from features.matchup import (
    add_opponent,
    add_team_matchup_features,
    validate_matchup_features,
)


DATA_FOLDER = Path("data")
DK_FOLDER = DATA_FOLDER / "dk_salaries"

ESPN_FILE = DATA_FOLDER / "nba_player_games_2026_27.csv"
MAPPING_FILE = DATA_FOLDER / "dk_espn_player_mapping.csv"

# Automatically select the latest DraftKings salary file
def select_salary_file():
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", default=None)
    parser.add_argument("--slate", default="Main")

    args = parser.parse_args()

    candidates = []

    for file in DK_FOLDER.glob("DKSalaries*.csv"):
        match = re.fullmatch(
            r"DKSalaries_(\d{4})_(\d{2})_(\d{2})_(Early|Main|Late)\.csv",
            file.name,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        year, month, day, slate = match.groups()
        file_date = pd.Timestamp(
            f"{year}-{month}-{day}"
        )

        if args.date and file_date != pd.Timestamp(args.date):
            continue

        if slate.lower() != args.slate.lower():
            continue

        candidates.append((file_date, file))

    if not candidates:
        raise FileNotFoundError(
            "No matching DraftKings salary file found."
        )

    slate_date, selected_file = max(
        candidates,
        key=lambda item: item[0]
    )


    return selected_file, slate_date


DK_FILE, SLATE_DATE = select_salary_file()



def main():
    print("DraftKings Player Analysis")
    print("=" * 40)
    print(f"Salary file: {DK_FILE.name}")

    # Load data
    dk = pd.read_csv(DK_FILE, dtype={"ID": str})
    mapping = pd.read_csv(
        MAPPING_FILE,
        dtype={"DK_ID": str, "ESPN_PLAYER_ID": str}
    )
    games = pd.read_csv(
        ESPN_FILE,
        dtype={"PLAYER_ID": str}
    )

    # Ensure statistical columns are numeric
    numeric_columns = [
        "MIN", "PTS", "REB", "AST", "DK_POINTS"
    ]

    for column in numeric_columns:
        games[column] = pd.to_numeric(
            games[column],
            errors="coerce"
        )

    # Exclude records without valid playing time
    # Keep these records in the original ESPN CSV
    games = games[games["MIN"].notna()].copy()

    # Convert dates and sort each player's games chronologically
    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"],
        errors="coerce"
    )

    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    )

    # Only use games played before the selected slate
    games = games[
        games["GAME_DATE"] < SLATE_DATE
    ].copy()

    print(f"\nSlate date: {SLATE_DATE.date()}")
    print(f"Historical player-game records: {len(games)}")

    # Load historical team games
    team_games = pd.read_csv(
        "data/nba_team_games_2026_27.csv",
        dtype={"GAME_ID": str}
    )

    team_games["GAME_DATE"] = pd.to_datetime(
        team_games["GAME_DATE"]
    )

    # Exclude games on or after the slate date
    team_games = team_games[
        team_games["GAME_DATE"] < SLATE_DATE
    ].copy()

    print(f"Historical team records: {len(team_games)}")

    team_features = calculate_team_features(team_games)

    print(f"Teams with historical features: {len(team_features)}")
    print(
        team_features[
            ["TEAM", "PACE_SEASON", "DEF_RATING_SEASON"]
        ].head().round(2).to_string(index=False)
    )


    # Connect DraftKings players to ESPN IDs
    players = dk.merge(
        mapping[["DK_ID", "ESPN_PLAYER_ID"]],
        left_on="ID",
        right_on="DK_ID",
        how="left",
        validate="one_to_one"
    )


    players = add_opponent(players)
    players = add_team_matchup_features(players, team_features)
    validate_matchup_features(players)
    print(
        players[
            ["Name", "MATCHUP_PACE_SEASON", "MATCHUP_PACE_LAST_3"]
        ].head(10).round(2).to_string(index=False)
        )

# ---------------------------------------------------------
# PLAYER FEATURE ENGINEERING
# Calculate historical scoring and playing-time statistics.
# Each feature module returns one row per ESPN player ID.
# Merge results into the DraftKings slate for analysis.
# All game history is filtered to dates before the slate.
# ---------------------------------------------------------

    # Calculate historical performance by player
    stats = calculate_scoring_features(games)

    # Merge performance into salary data
    players = players.merge(
        stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )

    # Calculate recent performance for each player
    recent_stats = calculate_rolling_scoring_features(games)

    # Merge rolling stats into salary data
    players = players.merge(
        recent_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )
    # Calculate minute state performance for each player
    minutes_stats = calculate_minutes_features(games)
    
    # Merge minutes stats into salary data
    players = players.merge(
        minutes_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one",
    )

    # Calculate momentum using short- and long-term exponential moving averages
    momentum_stats = calculate_momentum_features(games)

    players = players.merge(
        momentum_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one",
    )

    # Calculate fantasy points per $1,000 of salary
    players["VALUE_PER_1000"] = (
        players["AVG_DK_POINTS"] /
        (players["Salary"] / 1000)
    )

    # Sort by salary
    players = players.sort_values(
        "Salary", ascending=False
    )

    # Save analysis dataset
    output = DATA_FOLDER / (
        f"dk_player_analysis_{SLATE_DATE:%Y_%m_%d}_"
        f"{DK_FILE.stem.rsplit('_', 1)[-1]}.csv"
    )
    players.to_csv(output, index=False)

    print(f"\nDraftKings players: {len(players)}")
    print(
        f"Players with ESPN IDs: "
        f"{players['ESPN_PLAYER_ID'].notna().sum()}"
    )
    print(
        f"Players with game history: "
        f"{players['GAMES_PLAYED'].notna().sum()}"
    )

    print("\nTop 10 by salary:")
    print(
        players[
            ["Name", "Salary", "GAMES_PLAYED",
             "AVG_DK_POINTS", "VALUE_PER_1000"]
        ].head(10).round(2).to_string(index=False)
    )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
