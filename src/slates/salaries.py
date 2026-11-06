
import re
from pathlib import Path

import pandas as pd

from src.data_processing.slate import select_salary_file


SALARY_FOLDER = Path("data/dk_salaries")

REQUIRED_COLUMNS = {
    "ID", "Name", "Position", "Roster Position",
    "Salary", "Game Info", "TeamAbbrev",
}

GAME_PATTERN = re.compile(
    r"^([A-Z]{2,3})@([A-Z]{2,3})\s+"
    r"(\d{2}/\d{2}/\d{4})\s+"
    r"(\d{2}:\d{2}[AP]M)\s+ET$"
)


def parse_game_info(game_info, team):
    """Extract known pregame matchup information."""
    match = GAME_PATTERN.fullmatch(str(game_info).strip())

    if not match:
        raise ValueError(f"Invalid Game Info: {game_info}")

    away, home, date, time = match.groups()
    team = str(team).strip().upper()

    if team not in (away, home):
        raise ValueError(
            f"Team {team} is not in matchup {away}@{home}"
        )

    opponent = home if team == away else away

    return {
        "GAME_DATE": pd.to_datetime(date).date(),
        "GAME_TIME_ET": pd.to_datetime(
            f"{date} {time}", format="%m/%d/%Y %I:%M%p"
        ),
        "OPPONENT": opponent,
        "HOME_AWAY": "HOME" if team == home else "AWAY",
        "MATCHUP": f"{away}@{home}",
    }


def load_dk_salaries(path):
    """Load and normalize a DraftKings Classic salary file."""
    df = pd.read_csv(path, dtype={"ID": str})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["ID"].isna().any() or df["ID"].duplicated().any():
        raise ValueError("Missing or duplicate DraftKings IDs")

    df["Salary"] = pd.to_numeric(df["Salary"], errors="raise")

    if df["Salary"].isna().any() or (df["Salary"] <= 0).any():
        raise ValueError("Invalid salary values")

    df["TeamAbbrev"] = df["TeamAbbrev"].str.strip().str.upper()

    game_rows = [
        parse_game_info(info, team)
        for info, team in zip(df["Game Info"], df["TeamAbbrev"])
    ]

    games = pd.DataFrame(game_rows, index=df.index)
    df = pd.concat([df, games], axis=1)

    df["DK_ID"] = df["ID"]
    df["DK_NAME"] = df["Name"]
    df["TEAM"] = df["TeamAbbrev"]

    df["ELIGIBLE_POSITIONS"] = df["Roster Position"].str.split("/")

    if "Status" not in df:
        df["Status"] = ""

    df["STATUS"] = df["Status"].fillna("").astype(str).str.upper().str.strip()
    df["IS_OUT"] = df["STATUS"].isin(["O", "OUT"])
    df["IS_QUESTIONABLE"] = df["STATUS"].isin(["Q", "QUESTIONABLE"])

    if "AvgPointsPerGame" in df:
        df["DK_AVG_POINTS"] = pd.to_numeric(
            df["AvgPointsPerGame"], errors="coerce"
        )
    else:
        df["DK_AVG_POINTS"] = float("nan")

    return df


def main():
    salary_file, slate_date = select_salary_file(SALARY_FOLDER)
    df = load_dk_salaries(salary_file)

    print(f"Salary file: {salary_file.name}")
    print(f"Slate date: {slate_date.date()}")
    print(f"Players: {len(df)}")
    print(f"Games: {df['MATCHUP'].nunique()}")
    print(f"Out: {df['IS_OUT'].sum()}")
    print(f"Questionable: {df['IS_QUESTIONABLE'].sum()}")

    print("\nMatchups:")
    print("\n".join(sorted(df["MATCHUP"].unique())))


if __name__ == "__main__":
    main()
