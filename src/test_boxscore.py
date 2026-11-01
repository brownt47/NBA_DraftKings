
import requests
import pandas as pd

GAME_ID = "401909277"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

response = requests.get(URL, params={"event": GAME_ID}, timeout=30)
response.raise_for_status()

data = response.json()
records = []

for team_data in data["boxscore"]["players"]:
    team = team_data["team"]["abbreviation"]

    for stat_group in team_data["statistics"]:
        columns = stat_group["names"]

        for player in stat_group["athletes"]:
            if player.get("didNotPlay", False):
                continue

            stats = dict(zip(columns, player["stats"]))
            athlete = player["athlete"]

            record = {
                "GAME_ID": GAME_ID,
                "PLAYER_ID": athlete["id"],
                "PLAYER_NAME": athlete["displayName"],
                "TEAM": team,
                **stats,
            }

            records.append(record)

df = pd.DataFrame(records)

print(f"Players who played: {len(df)}")
print()
print(df.to_string(index=False))
