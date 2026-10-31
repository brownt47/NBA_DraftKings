import time
from pathlib import Path

import pandas as pd
import requests


ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/teams/{team}/roster"
)

OUTPUT_FILE = Path("data/espn_player_directory.csv")

session = requests.Session()


def get_roster(team):
    """Download one team's roster with retries."""

    url = ROSTER_URL.format(team=team)

    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as error:
            print(
                f"  {team}: attempt {attempt}/3 failed: {error}"
            )

            if attempt < 3:
                time.sleep(3 * attempt)

    raise RuntimeError(
        f"Unable to download roster for {team}"
    )

def build_player_directory():
    """Collect player IDs and names from all NBA rosters."""

    records = []

    for index, team in enumerate(TEAMS, start=1):
        data = get_roster(team)

        team_name = data["team"]["displayName"]
        athletes = data.get("athletes", [])

        print(
            f"{index}/{len(TEAMS)}: "
            f"{team_name} ({len(athletes)} players)"
        )

        for athlete in athletes:
            records.append({
                "PLAYER_ID": str(athlete["id"]),
                "PLAYER_NAME": athlete["displayName"],
                "TEAM": data["team"]["abbreviation"]
            })

        time.sleep(0.2)

    df = pd.DataFrame(records)

    # Prevent duplicate player IDs
    df = df.drop_duplicates(
        subset=["PLAYER_ID"]
    )

    df = df.sort_values(
        ["PLAYER_NAME"]
    ).reset_index(drop=True)

    return df

TEAMS = [
    "atl", "bos", "bkn", "cha", "chi", "cle",
    "dal", "den", "det", "gs", "hou", "ind",
    "lac", "lal", "mem", "mia", "mil", "min",
    "no", "ny", "okc", "orl", "phi", "phx",
    "por", "sac", "sa", "tor", "utah", "wsh"
]

def main():
    print("ESPN NBA Player Directory")
    print("=" * 40)

    df = build_player_directory()

    if df.empty:
        raise RuntimeError("No players found. CSV unchanged.")

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 40)
    print("DIRECTORY COMPLETE")
    print(f"Players: {len(df)}")
    print(f"Teams: {df['TEAM'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()