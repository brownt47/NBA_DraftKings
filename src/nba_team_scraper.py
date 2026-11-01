
import requests
import pandas as pd


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



if __name__ == "__main__":
    player_games = pd.read_csv(
        "data/nba_player_games_2026_27.csv"
    )

    game_ids = player_games["GAME_ID"].dropna().unique()

    print(f"Found {len(game_ids)} unique games.")

    records = []

    for i, game_id in enumerate(game_ids, start=1):
        try:
            teams = fetch_team_game(str(game_id))
            records.extend(teams)

            print(
                f"{i}/{len(game_ids)}: "
                f"Game {game_id}, {len(teams)} team records"
            )

        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"ERROR: Game {game_id}: {error}")

    if records:
        df = pd.DataFrame(records)

        output = "data/nba_team_games_2026_27.csv"

        df.to_csv(output, index=False)

        print(f"\nSaved {len(df)} records to {output}")
        print(f"Unique teams: {df['TEAM'].nunique()}")
        print(f"Unique games: {df['GAME_ID'].nunique()}")
