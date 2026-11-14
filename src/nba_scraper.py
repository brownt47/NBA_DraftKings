import argparse
from html import parser
import os
from sys import argv
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.dk_scoring import calculate_dk_points

# Configuration
SEASONS = {
    "2025-26": (
        date(2025, 10, 21),
        date(2026, 4, 12),
    ),
    "2026-27": (
        date(2026, 10, 20),
        None,
    ),
}

START_DATE, _ = SEASONS["2026-27"]
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
PARTICIPATION_OUTPUT_FILE = Path(
    "data/nba_player_participation_2026_27.csv"
)

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


def discover_games(start_date=START_DATE, end_date=None):
    """Find completed regular-season games within a date range."""
    if end_date is None:
        end_date = END_DATE

    games = {}
    current_date = start_date

    while current_date <= end_date:
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


            # Convert ESPN's UTC game timestamp to Eastern calendar date.
            game_timestamp = event.get("date")

            if not game_timestamp:
                raise ValueError(
                    f"Missing ESPN date for game {game_id}"
                )

            game_date = (
                pd.to_datetime(game_timestamp, utc=True)
                .tz_convert("America/New_York")
                .date()
                .isoformat()
            )

            games[game_id] = {
                "GAME_ID": game_id,
                "GAME_DATE": game_date,
                "MATCHUP": event.get("name", ""),
            }


            count += 1

        print(f"{current_date}: {count} completed games")
        current_date += timedelta(days=1)

    return games


def discover_scheduled_games(game_date):
    """Retrieve scheduled NBA games, including those not yet completed."""
    date_string = game_date.strftime("%Y%m%d")

    data = get_json(
        SCOREBOARD_URL,
        {"dates": date_string, "limit": 100},
    )

    games = []

    for event in data.get("events", []):
        if event.get("season", {}).get("type") != 2:
            continue

        competition = event["competitions"][0]
        competitors = competition["competitors"]

        home = next(
            team for team in competitors
            if team["homeAway"] == "home"
        )
        away = next(
            team for team in competitors
            if team["homeAway"] == "away"
        )

        games.append({
            "GAME_ID": str(event["id"]),
            "GAME_DATE": game_date.isoformat(),
            "MATCHUP": event.get("name", ""),
            "HOME_TEAM": home["team"]["abbreviation"],
            "AWAY_TEAM": away["team"]["abbreviation"],
        })

    return games


def split_shooting(value):
    """Convert a shooting stat such as '8-18' to (8, 18)."""
    made, attempted = str(value).split("-")
    return int(made), int(attempted)

# ---------------------------------------------------------
# PARSE PLUS/MINUS
# Convert ESPN plus/minus to a consistent numeric value.
# ---------------------------------------------------------
def parse_plus_minus(value):
    if value is None or str(value).strip() in ("", "--"):
        return None

    return int(value)

def parse_boxscore(game, include_participation=False):
    """Convert one ESPN box score into player and participation records."""
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
    participation_records = []

    for team_data in teams:
        team = team_data["team"]["abbreviation"]

        for stat_group in team_data["statistics"]:
            columns = stat_group["names"]

            for player in stat_group["athletes"]:
                athlete = player["athlete"]

                did_not_play = bool(player.get("didNotPlay", False))

                if include_participation and athlete.get("id"):
                    participation_records.append({
                        "GAME_ID": str(game["GAME_ID"]),
                        "GAME_DATE": game["GAME_DATE"],
                        "PLAYER_ID": str(athlete["id"]),
                        "PLAYER_NAME": athlete["displayName"],
                        "TEAM": team,
                        "PLAYED": not did_not_play,
                        "DID_NOT_PLAY": did_not_play,
                        "DNP_REASON": (
                            player.get("reason") if did_not_play else None
                        ),
                        "ACTIVE": player.get("active"),
                        "STARTER": player.get("starter"),
                        "MIN": (
                            player.get("stats", [None])[0]
                            if player.get("stats")
                            else None
                        ),
                    })

                if did_not_play:
                    continue

                if player["stats"][0] == "--" and not player["athlete"].get("id"):
                    print(
                        f"  Skipping incomplete ESPN athlete record: "
                        f"{player['athlete'].get('shortName', 'Unknown')} "
                        f"(game {game['GAME_ID']})"
                    )
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
                    "PLUS_MINUS": parse_plus_minus(stats.get("+/-")),
                }

                records.append(record)

    if not records:
        raise ValueError(
            f"No player records: {game['GAME_ID']}"
        )

    if include_participation:
        return records, participation_records

    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download ESPN NBA player box scores."
    )

    parser.add_argument(
        "--season",
        choices=SEASONS.keys(),
        default="2026-27",
    )

    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
    )

    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
    )

    parser.add_argument(
        "--participation",
        action="store_true",
        help="Collect ESPN player participation and DNP records.",
    )

    args = parser.parse_args([] if argv is None else argv)

    season_start, season_end = SEASONS[args.season]

    start_date = args.start_date or season_start
    end_date = args.end_date or season_end or date.today()

    if start_date > end_date:
        parser.error("Start date cannot be after end date.")

    output_file = Path(
        f"data/nba_player_games_{args.season.replace('-', '_')}.csv"
    )

    if args.season == "2026-27":
        output_file = OUTPUT_FILE

    participation_file = Path(
        f"data/nba_player_participation_{args.season.replace('-', '_')}.csv"
    )

    if args.season == "2026-27":
        participation_file = PARTICIPATION_OUTPUT_FILE

    print(f"ESPN NBA {args.season} Player Game Scraper")
    print(f"Date range: {start_date} through {end_date}")
    print(f"Output: {output_file}")
    print(f"Participation Output: {participation_file}")
    print("=" * 40)

    # Load existing data if available
    if output_file.exists():
        existing_df = pd.read_csv(
            output_file,
            dtype={"GAME_ID": str, "PLAYER_ID": str}
        )

        if existing_df.empty:
            refresh_start = start_date
            print("\nExisting CSV is empty. Full-season download.")

        elif args.season == "2025-26" and args.start_date is not None:
            refresh_start = start_date

        else:
            latest_saved_date = date.fromisoformat(
                existing_df["GAME_DATE"].max()
            )

            refresh_start = max(
                start_date,
                latest_saved_date - timedelta(days=REFRESH_DAYS - 1)
            )

            print(
                f"\nIncremental update: "
                f"refreshing games from {refresh_start}"
            )

    else:
        existing_df = pd.DataFrame()
        refresh_start = start_date

        print("\nNo existing CSV. Full-season download.")

    refresh_start_stats = refresh_start

    if args.participation:
        if participation_file.exists():
            participation_df = pd.read_csv(
                participation_file,
                dtype={"GAME_ID": str, "PLAYER_ID": str},
            )

            if participation_df.empty:
                participation_refresh_start = start_date
                print("\nParticipation CSV is empty. Full backfill.")
            else:
                latest_participation_date = date.fromisoformat(
                    participation_df["GAME_DATE"].max()
                )

                participation_refresh_start = max(
                    start_date,
                    latest_participation_date
                    - timedelta(days=REFRESH_DAYS - 1),
                )

                print(
                    "\nParticipation incremental update: "
                    f"refreshing from {participation_refresh_start}"
                )
        else:
            participation_df = pd.DataFrame()
            participation_refresh_start = start_date
            print("\nNo participation CSV. Full backfill.")

        refresh_start = min(
            refresh_start,
            participation_refresh_start,
        )




    print("\nDiscovering completed games...")
    print(f"DEBUG: refresh_start = {refresh_start}")
    print(f"DEBUG: end_date = {end_date}")
    games = discover_games(refresh_start, end_date)

    print(f"\nFound {len(games)} completed games.")

    if not games:
        print("No completed games to refresh. CSV unchanged.")
        return

    all_records = []
    all_participation_records = []

    print("\nDownloading player box scores...")

    for index, game in enumerate(games.values(), start=1):

        if args.participation:
            records, participation_records = parse_boxscore(
                game,
                include_participation=True,
            )
            all_participation_records.extend(participation_records)
        else:
            records = parse_boxscore(game)

        # During a historical participation backfill, preserve
        # older player-statistics records already in the CSV.
        if (
            not args.participation
            or date.fromisoformat(str(game["GAME_DATE"]))
            >= refresh_start_stats
        ):
            all_records.extend(records)

        print(
            f"{index}/{len(games)}: "
            f"{game['GAME_DATE']} "
            f"{game['MATCHUP']} "
            f"({len(records)} players)"
        )

        time.sleep(0.1)

    
    # Build DataFrame from newly downloaded games
    new_df = pd.DataFrame(all_records)

    # Remove only statistics records we actually refreshed.
    # Historical games downloaded solely for participation
    # must retain their existing player statistics.
    if not existing_df.empty:
        refreshed_game_ids = {
            str(game["GAME_ID"])
            for game in games.values()
            if (
                not args.participation
                or date.fromisoformat(str(game["GAME_DATE"]))
                >= refresh_start_stats
            )
        }

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

    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary_file = output_file.with_suffix(".tmp")

    try:
        df.to_csv(temporary_file, index=False)
        os.replace(temporary_file, output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


    # Save player participation records separately from statistics.
    if args.participation:
        new_participation_df = pd.DataFrame(
            all_participation_records
        )

        if not participation_df.empty:
            refreshed_participation_game_ids = {
                str(game["GAME_ID"])
                for game in games.values()
            }

            participation_df = participation_df[
                ~participation_df["GAME_ID"].isin(
                    refreshed_participation_game_ids
                )
            ]

            combined_participation_df = pd.concat(
                [participation_df, new_participation_df],
                ignore_index=True,
            )
        else:
            combined_participation_df = new_participation_df

        if combined_participation_df.empty:
            raise RuntimeError(
                "No participation records. CSV unchanged."
            )

        duplicates = combined_participation_df.duplicated(
            subset=["GAME_ID", "PLAYER_ID"]
        )

        if duplicates.any():
            raise RuntimeError(
                "Duplicate participation records detected."
            )

        combined_participation_df = (
            combined_participation_df.sort_values(
                ["GAME_DATE", "GAME_ID", "TEAM", "PLAYER_NAME"]
            ).reset_index(drop=True)
        )

        participation_file.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary_participation_file = (
            participation_file.with_suffix(".tmp")
        )

        try:
            combined_participation_df.to_csv(
                temporary_participation_file,
                index=False,
            )
            os.replace(
                temporary_participation_file,
                participation_file,
            )
        finally:
            if temporary_participation_file.exists():
                temporary_participation_file.unlink()

        print(
            "Participation records saved: "
            f"{len(combined_participation_df):,}"
        )
        print(f"Participation saved to: {participation_file}")


    print("\n" + "=" * 40)
    print("SCRAPE COMPLETE")
    print(f"Games refreshed: {len(games)}")
    print(f"Total games in dataset: {df['GAME_ID'].nunique()}")
    print(f"Player-game records: {len(df):,}")
    print(f"Unique players: {df['PLAYER_ID'].nunique()}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
