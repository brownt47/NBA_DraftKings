
"""Build and maintain the ESPN NBA player directory."""

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
import requests


ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams/{team}/roster"
)

ATHLETE_URL = (
    "https://site.web.api.espn.com/apis/common/v3/"
    "sports/basketball/nba/athletes/{player_id}"
)

OUTPUT_FILE = Path("data/espn_player_directory.csv")
HISTORY_FILE = Path("data/nba_player_games_2025_26.csv")

COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM",
    "POSITION",
    "POSITION_NAME",
]

TEAMS = [
    "atl", "bos", "bkn", "cha", "chi", "cle",
    "dal", "den", "det", "gs", "hou", "ind",
    "lac", "lal", "mem", "mia", "mil", "min",
    "no", "ny", "okc", "orl", "phi", "phx",
    "por", "sac", "sa", "tor", "utah", "wsh",
]

session = requests.Session()


def get_json(url):
    """Download JSON with retries."""
    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            print(f"Attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                time.sleep(3 * attempt)

    return None


def get_roster(team):
    """Download one team's current roster."""
    data = get_json(ROSTER_URL.format(team=team))
    if data is None:
        raise RuntimeError(f"Unable to download roster for {team}")
    return data


def build_player_directory():
    """Collect current NBA roster players."""
    records = []

    for index, team in enumerate(TEAMS, start=1):
        data = get_roster(team)
        team_data = data["team"]
        athletes = data.get("athletes", [])

        print(
            f"{index}/{len(TEAMS)}: "
            f"{team_data['displayName']} ({len(athletes)} players)"
        )

        for athlete in athletes:
            position = athlete.get("position") or {}

            records.append({
                "PLAYER_ID": str(athlete["id"]),
                "PLAYER_NAME": athlete["displayName"],
                "TEAM": team_data["abbreviation"],
                "POSITION": position.get("abbreviation"),
                "POSITION_NAME": position.get("displayName"),
            })

        time.sleep(0.2)

    return (
        pd.DataFrame(records, columns=COLUMNS)
        .drop_duplicates(subset=["PLAYER_ID"])
        .sort_values("PLAYER_NAME")
        .reset_index(drop=True)
    )


def load_directory():
    """Load existing records without losing player IDs."""
    if not OUTPUT_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    return pd.read_csv(
        OUTPUT_FILE,
        dtype={"PLAYER_ID": str},
    )[COLUMNS]


def save_directory(df):
    """Back up and save the combined directory."""
    df = (
        df[COLUMNS]
        .drop_duplicates(subset=["PLAYER_ID"])
        .sort_values("PLAYER_NAME")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        backup = OUTPUT_FILE.with_suffix(".csv.bak")
        shutil.copy2(OUTPUT_FILE, backup)
        print(f"Backup: {backup}")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDIRECTORY COMPLETE")
    print(f"Players: {len(df)}")
    print(f"Players with positions: {df['POSITION'].notna().sum()}")
    print(f"Saved to: {OUTPUT_FILE}")


def refresh_rosters():
    """Refresh current players while retaining historical records."""
    current = build_player_directory()
    historical = load_directory()

    combined = pd.concat(
        [current, historical],
        ignore_index=True,
    )

    save_directory(combined)


def backfill_history():
    """Retrieve ESPN positions for missing historical players."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(HISTORY_FILE)

    directory = load_directory()

    if directory.empty:
        raise RuntimeError(
            "Existing directory is empty. Run roster refresh first."
        )

    games = pd.read_csv(
        HISTORY_FILE,
        dtype={"PLAYER_ID": str},
    )

    historical_players = (
        games[["PLAYER_ID", "PLAYER_NAME", "TEAM"]]
        .drop_duplicates(subset=["PLAYER_ID"])
    )

    known_ids = set(directory["PLAYER_ID"])

    missing = historical_players[
        ~historical_players["PLAYER_ID"].isin(known_ids)
    ]

    print(f"Missing historical players: {len(missing)}")

    new_records = []
    failures = []

    for index, player in enumerate(
        missing.itertuples(index=False),
        start=1,
    ):
        player_id = player.PLAYER_ID

        data = get_json(
            ATHLETE_URL.format(player_id=player_id)
        )

        if data is None:
            failures.append((player_id, player.PLAYER_NAME))
            continue

        athlete = data.get("athlete") or {}
        position = athlete.get("position") or {}

        abbreviation = position.get("abbreviation")
        position_name = position.get("displayName")

        if not abbreviation:
            failures.append((player_id, player.PLAYER_NAME))
            continue

        new_records.append({
            "PLAYER_ID": player_id,
            "PLAYER_NAME": athlete.get(
                "displayName", player.PLAYER_NAME
            ),
            "TEAM": player.TEAM,
            "POSITION": abbreviation,
            "POSITION_NAME": position_name,
        })

        print(
            f"{index}/{len(missing)}: "
            f"{player.PLAYER_NAME} -> {abbreviation}"
        )

        time.sleep(0.2)

    if new_records:
        additions = pd.DataFrame(new_records, columns=COLUMNS)
        combined = pd.concat(
            [directory, additions],
            ignore_index=True,
        )
        save_directory(combined)
    else:
        print("No new records retrieved. Directory unchanged.")

    print(f"\nSuccessfully backfilled: {len(new_records)}")
    print(f"Failed lookups: {len(failures)}")

    for player_id, name in failures:
        print(f"  {player_id}: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Retrieve positions for missing 2025-26 players",
    )
    args = parser.parse_args()

    if args.backfill_history:
        backfill_history()
    else:
        refresh_rosters()


if __name__ == "__main__":
    main()
