import os
from pathlib import Path

import requests
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PLAYER_FILE = Path("data/nba_player_games_2026_27.csv")
OUTPUT_FILE = Path("data/nba_team_games_2026_27.csv")



def fetch_team_game(game_id):
    """Return two team-level records from one completed NBA game."""

    url = (
        "https://site.api.espn.com/apis/site/v2/"
        f"sports/basketball/nba/summary?event={game_id}"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()

    competition = data["header"]["competitions"][0]
    competitors = competition["competitors"]

    if not competition["status"]["type"]["completed"]:
        return []

    periods = len(competitors[0].get("linescores", []))

    if periods < 4:
        return []

    game_minutes = 48 + 5 * (periods - 4)

    teams = []

    for competitor in competitors:
        team_info = competitor["team"]
        team_id = str(team_info["id"])

        team_stats = next(
            t["statistics"]
            for t in data["boxscore"]["teams"]
            if str(t["team"]["id"]) == team_id
        )

        stats = {
            s["name"]: s["displayValue"]
            for s in team_stats
        }

        def made_attempted(value):
            made, attempted = value.split("-")
            return int(made), int(attempted)

        fgm, fga = made_attempted(
            stats["fieldGoalsMade-fieldGoalsAttempted"]
        )
        ftm, fta = made_attempted(
            stats["freeThrowsMade-freeThrowsAttempted"]
        )

        team = {
            "GAME_ID": str(game_id),
            "GAME_DATE": competition["date"][:10],
            "TEAM": team_info["abbreviation"],
            "HOME_AWAY": competitor["homeAway"],
            "PTS": int(competitor["score"]),
            "FGM": fgm,
            "FGA": fga,
            "FTM": ftm,
            "FTA": fta,
            "OREB": int(stats["offensiveRebounds"]),
            "TOV": int(stats["totalTurnovers"]),
            "GAME_MINUTES": game_minutes,
        }

        team["POSS_EST_RAW"] = (
            team["FGA"]
            + 0.44 * team["FTA"]
            - team["OREB"]
            + team["TOV"]
        )

        teams.append(team)

    if len(teams) != 2:
        return []

    # Use the average of both teams' possession estimates.
    possessions = sum(
        team["POSS_EST_RAW"] for team in teams
    ) / 2

    for team, opponent in zip(teams, reversed(teams)):
        team["OPPONENT"] = opponent["TEAM"]
        team["OPP_PTS"] = opponent["PTS"]
        team["POSS_EST"] = possessions
        team["PACE_EST"] = possessions * 48 / game_minutes
        team["OFF_RATING_EST"] = (
            100 * team["PTS"] / possessions
            if possessions > 0 else float("nan")
        )
        team["DEF_RATING_EST"] = (
            100 * team["OPP_PTS"] / possessions
            if possessions > 0 else float("nan")
        )

    return teams




# ---------------------------------------------------------
# MAIN SCRAPER
# Download team statistics for completed NBA games.
# ---------------------------------------------------------

def main():

    # ---------------------------------------------------------
    # LOAD PLAYER GAME DATA
    # ---------------------------------------------------------

    player_games = pd.read_csv(
        PLAYER_FILE,
        dtype={"GAME_ID": str}
    )


    # ---------------------------------------------------------
    # RECENT GAME REFRESH WINDOW
    # Refresh games from the latest three calendar dates.
    # ---------------------------------------------------------

    player_games["GAME_DATE"] = pd.to_datetime(
        player_games["GAME_DATE"]
    )

    latest_date = player_games["GAME_DATE"].max()

    refresh_start = latest_date - pd.Timedelta(days=2)

    recent_ids = set(
        player_games.loc[
            player_games["GAME_DATE"] >= refresh_start,
            "GAME_ID"
        ].dropna()
    )


    # ---------------------------------------------------------
    # IDENTIFY GAMES TO DOWNLOAD
    # Skip games already saved in the team dataset.
    # ---------------------------------------------------------

    game_ids = player_games["GAME_ID"].dropna().unique()

    print(f"Found {len(game_ids)} unique games.")

    if OUTPUT_FILE.exists():

        existing_df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"GAME_ID": str}
        )

    else:
        existing_df = pd.DataFrame()

    # Only consider games complete when both team records exist
    if not existing_df.empty:

        game_counts = existing_df.groupby("GAME_ID").size()

        completed_ids = set(
            game_counts[game_counts == 2].index
        )

    else:
        completed_ids = set()


    game_ids = [
        game_id
        for game_id in game_ids
        if game_id not in completed_ids
        or game_id in recent_ids
    ]


    print(f"Games needing download: {len(game_ids)}")

    records = []


    for i, game_id in enumerate(game_ids, start=1):

        try:

            teams = fetch_team_game(str(game_id))

            # Only save complete games with two distinct teams
            if (
                len(teams) != 2
                or len({team["TEAM"] for team in teams}) != 2
                or any(
                    str(team["GAME_ID"]) != str(game_id)
                    for team in teams
                )
            ):
                print(
                    f"WARNING: Game {game_id} returned "
                    f"{len(teams)} invalid/incomplete team records. "
                    "Skipping."
                )
                continue

            records.extend(teams)


            print(
                f"{i}/{len(game_ids)}: "
                f"Game {game_id}, {len(teams)} team records"
            )

        except (
            requests.RequestException,
            KeyError,
            ValueError
        ) as error:

            print(f"ERROR: Game {game_id}: {error}")


    # ---------------------------------------------------------
    # COMBINE EXISTING AND NEW TEAM RECORDS
    # Preserve historical games and replace incomplete games.
    # ---------------------------------------------------------

    if not records:
        print("\nNo new team records downloaded. CSV unchanged.")
        return

    new_df = pd.DataFrame(records)

    if not existing_df.empty:

        # Remove old records for games being refreshed
        refreshed_ids = set(new_df["GAME_ID"])

        existing_df = existing_df[
            ~existing_df["GAME_ID"].isin(refreshed_ids)
        ]

        df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )

    else:
        df = new_df

    # ---------------------------------------------------------
    # VALIDATE TEAM RECORDS
    # ---------------------------------------------------------

    duplicates = df.duplicated(
        subset=["GAME_ID", "TEAM"]
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate team-game records detected. "
            "CSV unchanged."
        )

    game_counts = df.groupby("GAME_ID").size()

    if not game_counts.eq(2).all():
        raise RuntimeError(
            "Incomplete team-game records detected. "
            "CSV unchanged."
        )

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # SAFELY SAVE TEAM DATA
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    try:
        df.to_csv(temporary_file, index=False)
        os.replace(temporary_file, OUTPUT_FILE)

    finally:
        if temporary_file.exists():
            temporary_file.unlink()

    print("\n" + "=" * 40)
    print("TEAM SCRAPE COMPLETE")
    print(f"New team records: {len(new_df)}")
    print(f"Total games: {df['GAME_ID'].nunique()}")
    print(f"Total team records: {len(df):,}")
    print(f"Unique teams: {df['TEAM'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")

# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
