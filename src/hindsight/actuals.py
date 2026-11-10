
"""Prepare historical DraftKings slates using actual ESPN results."""

from pathlib import Path

import pandas as pd
from src.features.matchup import DK_TO_ESPN

SALARY_DIR = Path("data/dk_salaries")
MAPPING_FILE = Path("data/dk_espn_player_mapping.csv")
GAMES_FILE = Path("data/nba_player_games_2026_27.csv")


def prepare_actuals(date, salary_dir=SALARY_DIR,
                    mapping_file=MAPPING_FILE, games_file=GAMES_FILE):
    """Return optimizer-ready players and a data-coverage report."""

    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    salary_file = Path(salary_dir) / (
        f"DKSalaries_{date.replace('-', '_')}_Main.csv"
    )

    for path in (salary_file, Path(mapping_file), Path(games_file)):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")

    salary = pd.read_csv(salary_file)
    mapping = pd.read_csv(mapping_file)
    games = pd.read_csv(games_file)

    salary["ID"] = pd.to_numeric(salary["ID"], errors="raise")
    mapping["DK_ID"] = pd.to_numeric(mapping["DK_ID"], errors="raise")
    mapping["ESPN_PLAYER_ID"] = pd.to_numeric(
        mapping["ESPN_PLAYER_ID"], errors="raise"
    )
    games["PLAYER_ID"] = pd.to_numeric(
        games["PLAYER_ID"], errors="raise"
    )

    actuals = games.loc[
        games["GAME_DATE"].eq(date),
        ["PLAYER_ID", "TEAM", "DK_POINTS"],
    ].copy()

    # Keep team information to detect incorrect player joins.
    joined = salary.merge(
        mapping[["DK_ID", "ESPN_PLAYER_ID"]],
        left_on="ID",
        right_on="DK_ID",
        how="left",
        validate="many_to_one",
    )

    joined = joined.merge(
        actuals,
        left_on="ESPN_PLAYER_ID",
        right_on="PLAYER_ID",
        how="left",
        validate="many_to_one",
    )

    team_mismatch = (
        joined["DK_POINTS"].notna()
        & joined["TeamAbbrev"]
            .replace(DK_TO_ESPN)
            .ne(joined["TEAM"])
    )
    if team_mismatch.any():
        names = joined.loc[team_mismatch, "Name"].tolist()
        raise ValueError(f"Team mismatch for players: {names}")

    report = {
        "date": date,
        "salary_players": len(salary),
        "mapped_players": int(joined["ESPN_PLAYER_ID"].notna().sum()),
        "players_with_results": int(joined["DK_POINTS"].notna().sum()),
        "players_without_results": int(joined["DK_POINTS"].isna().sum()),
        "games_with_results": int(
            games.loc[games["GAME_DATE"].eq(date), "GAME_ID"].nunique()
        ),
    }

    players = joined.loc[joined["DK_POINTS"].notna()].copy()

    players = players.rename(columns={
        "ID": "DK_ID_FINAL",
        "Name": "DK_NAME",
        "Salary": "SALARY",
        "TeamAbbrev": "TEAM_DK",
    })

    players["DK_ID"] = players["DK_ID_FINAL"]
    players["ELIGIBLE_POSITIONS"] = (
        players["Roster Position"]
        .fillna("")
        .str.split("/")
    )
    players["PROJECTED_DK_POINTS"] = pd.to_numeric(
        players["DK_POINTS"], errors="raise"
    )
    players["OPTIMIZATION_SCORE"] = players["PROJECTED_DK_POINTS"]
    players["OPTIMIZER_ELIGIBLE"] = True

    if players.empty:
        raise ValueError(f"No actual player results for {date}.")

    return players, report
