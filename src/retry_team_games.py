
import pandas as pd
from nba_team_scraper import fetch_team_game

path = "data/nba_team_games_2026_27.csv"

df = pd.read_csv(path, dtype={"GAME_ID": str})

missing_ids = ["401909088"]

records = []

for game_id in missing_ids:
    try:
        teams = fetch_team_game(game_id)
        records.extend(teams)
        print(f"{game_id}: {len(teams)} records")
    except Exception as error:
        print(f"{game_id}: ERROR: {error}")

if records:
    new_df = pd.DataFrame(records)

    df = pd.concat([df, new_df], ignore_index=True)

    df = df.drop_duplicates(
        subset=["GAME_ID", "TEAM"],
        keep="last"
    )

    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM"])

    df.to_csv(path, index=False)

print(f"\nTotal records: {len(df)}")
print(f"Unique games: {df['GAME_ID'].nunique()}")
