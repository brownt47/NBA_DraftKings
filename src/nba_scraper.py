
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from dk_scoring import calculate_dk_points

# Configuration
START_DATE = date(2026, 10, 20)
END_DATE = date.today()

# Number of recent days to refresh for stat corrections
REFRESH_DAYS = 3

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/scoreboard"
)

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/summary"
)

OUTPUT_FILE = Path("data/nba_player_games_2026_27.csv")

session = requests.Session()


def get_json(url, params):
    """Retrieve ESPN JSON, retrying temporary failures."""
    for attempt in range(1, 4):
        try:
            response = session.get(
                url, params=params, timeout=30
            )
            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as error:
            print(f"  Attempt {attempt}/3 failed: {error}")

            if attempt < 3:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Failed after 3 attempts: {params}")


def discover_games(start_date=START_DATE):
    """Find completed regular-season games from start_date."""
    games = {}
    current_date = start_date

    while current_date <= END_DATE:
        date_string = current_date.strftime("%Y%m%d")

        data = get_json(
            SCOREBOARD_URL,
            {"dates": date_string, "limit": 100}
        )

        count = 0

        for event in data.get("events", []):
            if event.get("season", {}).get("type") != 2:
                continue

            if not event.get("status", {}).get(
                "type", {}
            ).get("completed", False):
                continue

            game_id = event["id"]

            games[game_id] = {
                "GAME_ID": game_id,
                "GAME_DATE": current_date.isoformat(),
                "MATCHUP": event.get("name", ""),
            }

            count += 1

        print(f"{current_date}: {count} completed games")
        current_date += timedelta(days=1)

    return games


def split_shooting(value):
    """Convert a shooting stat such as '8-18' to (8, 18)."""
    made, attempted = str(value).split("-")
    return int(made), int(attempted)


def parse_boxscore(game):
    """Convert one ESPN box score into player records."""
    data = get_json(
        SUMMARY_URL,
        {"event": game["GAME_ID"]}
    )

    teams = data["boxscore"]["players"]

    if len(teams) != 2:
        raise ValueError(
            f"Unexpected box score: {game['GAME_ID']}"
        )

    records = []

    for team_data in teams:
        team = team_data["team"]["abbreviation"]

        for stat_group in team_data["statistics"]:
            columns = stat_group["names"]

            for player in stat_group["athletes"]:
                if player.get("didNotPlay", False):
                    continue

                stats = dict(zip(columns, player["stats"]))
                athlete = player["athlete"]

                fgm, fga = split_shooting(stats["FG"])
                fg3m, fg3a = split_shooting(stats["3PT"])
                ftm, fta = split_shooting(stats["FT"])

                record = {
                    **game,
                    "PLAYER_ID": athlete["id"],
                    "PLAYER_NAME": athlete["displayName"],
                    "TEAM": team,
                    "MIN": stats["MIN"],
                    "PTS": int(stats["PTS"]),
                    "FGM": fgm,
                    "FGA": fga,
                    "FG3M": fg3m,
                    "FG3A": fg3a,
                    "FTM": ftm,
                    "FTA": fta,
                    "REB": int(stats["REB"]),
                    "AST": int(stats["AST"]),
                    "TOV": int(stats["TO"]),
                    "STL": int(stats["STL"]),
                    "BLK": int(stats["BLK"]),
                    "OREB": int(stats["OREB"]),
                    "DREB": int(stats["DREB"]),
                    "PF": int(stats["PF"]),
                    "PLUS_MINUS": stats["+/-"],
                }

                records.append(record)

    if not records:
        raise ValueError(
            f"No player records: {game['GAME_ID']}"
        )

    return records


def main():
    print("ESPN NBA 2026-27 Player Game Scraper")
    print("=" * 40)

    # Load existing data if available
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"GAME_ID": str, "PLAYER_ID": str}
        )

        # Refresh recent games to capture stat corrections
        refresh_start = max(
            START_DATE,
            END_DATE - timedelta(days=REFRESH_DAYS - 1)
        )

        print(
            f"\nIncremental update: "
            f"refreshing games from {refresh_start}"
        )

    else:
        existing_df = pd.DataFrame()
        refresh_start = START_DATE

        print("\nNo existing CSV. Full-season download.")

    print("\nDiscovering completed games...")
    games = discover_games(refresh_start)

    print(f"\nFound {len(games)} completed games.")

    if not games:
        print("No completed games to refresh. CSV unchanged.")
        return

    all_records = []

    print("\nDownloading player box scores...")

    for index, game in enumerate(games.values(), start=1):
        records = parse_boxscore(game)
        all_records.extend(records)

        print(
            f"{index}/{len(games)}: "
            f"{game['GAME_DATE']} "
            f"{game['MATCHUP']} "
            f"({len(records)} players)"
        )

        time.sleep(0.1) # Be polite to ESPN's servers

    
    # Build DataFrame from newly downloaded games
    new_df = pd.DataFrame(all_records)

    # Remove old records for games being refreshed
    if not existing_df.empty:
        refreshed_game_ids = set(games.keys())

        existing_df = existing_df[
            ~existing_df["GAME_ID"].isin(refreshed_game_ids)
        ]

        # Combine historical and refreshed records
        df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )
    else:
        df = new_df

    # Recalculate DraftKings points for all records
    df = calculate_dk_points(df)


    if df.empty:
        raise RuntimeError("No player data. CSV unchanged.")

    duplicates = df.duplicated(
        subset=["GAME_ID", "PLAYER_ID"]
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate player-game records detected. "
            "CSV unchanged."
        )

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM", "PLAYER_NAME"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    try:
        df.to_csv(temporary_file, index=False)
        os.replace(temporary_file, OUTPUT_FILE)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()

    print("\n" + "=" * 40)
    print("SCRAPE COMPLETE")
    print(f"Games refreshed: {len(games)}")
    print(f"Total games in dataset: {df['GAME_ID'].nunique()}")
    print(f"Player-game records: {len(df):,}")
    print(f"Unique players: {df['PLAYER_ID'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
