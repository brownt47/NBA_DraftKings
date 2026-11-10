
"""Validate historical DraftKings Main slate box-score coverage.

Run:
    python -m src.hindsight.validation --date 2026-11-08
"""

import argparse
from pathlib import Path

import pandas as pd

from src.features.matchup import DK_TO_ESPN


SALARY_DIR = Path("data/dk_salaries")
GAMES_FILE = Path("data/nba_player_games_2026_27.csv")

# ESPN minutes are currently stored as rounded whole minutes.
MIN_MINUTES = 470
MAX_MINUTES = 490


def validate_slate(date, salary_dir=SALARY_DIR, games_file=GAMES_FILE):
    """Validate game coverage and approximate player minutes."""

    date = pd.Timestamp(date).strftime("%Y-%m-%d")

    salary_file = Path(salary_dir) / (
        f"DKSalaries_{date.replace('-', '_')}_Main.csv"
    )

    if not salary_file.exists():
        raise FileNotFoundError(f"Missing salary file: {salary_file}")

    if not Path(games_file).exists():
        raise FileNotFoundError(f"Missing games file: {games_file}")

    salary = pd.read_csv(salary_file)
    games = pd.read_csv(games_file)

    required_salary = {"Game Info", "TeamAbbrev"}
    required_games = {
        "GAME_ID", "GAME_DATE", "MATCHUP", "TEAM",
        "PLAYER_ID", "MIN", "DK_POINTS",
    }

    if not required_salary.issubset(salary.columns):
        raise ValueError("Salary file missing required columns.")

    if not required_games.issubset(games.columns):
        raise ValueError("Games file missing required columns.")

    slate_games = (
        salary["Game Info"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    day_games = games.loc[games["GAME_DATE"].eq(date)].copy()
    day_games["MIN_NUMERIC"] = pd.to_numeric(
        day_games["MIN"], errors="coerce"
    )

    records = []

    for game_info in slate_games:
        matchup = game_info.split()[0]

        if "@" not in matchup:
            raise ValueError(f"Invalid Game Info: {game_info}")

        away_dk, home_dk = matchup.split("@", 1)
        away = DK_TO_ESPN.get(away_dk, away_dk)
        home = DK_TO_ESPN.get(home_dk, home_dk)

        matching = day_games.loc[
            day_games["TEAM"].isin([away, home])
        ].copy()

        game_ids = matching["GAME_ID"].dropna().unique()

        if len(game_ids) != 1:
            records.append({
                "MATCHUP": matchup,
                "STATUS": "FAIL",
                "PLAYERS": len(matching),
                "TOTAL_MINUTES": None,
                "DETAIL": "Missing or ambiguous game ID",
            })
            continue

        game_id = game_ids[0]
        matching = matching.loc[matching["GAME_ID"].eq(game_id)]

        minutes = matching["MIN_NUMERIC"].sum()
        missing_minutes = matching["MIN_NUMERIC"].isna().sum()
        missing_points = matching["DK_POINTS"].isna().sum()

        if missing_minutes or missing_points:
            status = "FAIL"
            detail = "Missing minutes or DK points"
        elif not MIN_MINUTES <= minutes <= MAX_MINUTES:
            status = "WARNING"
            detail = "Minutes outside expected regulation range"
        else:
            status = "PASS"
            detail = "Game coverage and minutes consistent"

        records.append({
            "MATCHUP": matchup,
            "STATUS": status,
            "PLAYERS": len(matching),
            "TOTAL_MINUTES": minutes,
            "DETAIL": detail,
        })

    report = pd.DataFrame(records)

    if report.empty:
        raise ValueError(f"No DraftKings Main slate games for {date}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate historical NBA Main slate data."
    )
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    report = validate_slate(args.date)

    print(f"\nSLATE VALIDATION: {args.date}")
    print(report.to_string(index=False))

    print("\nSUMMARY")
    print(report["STATUS"].value_counts().to_string())

    if (report["STATUS"] == "FAIL").any():
        print("\nValidation failed. Review missing game data.")
        raise SystemExit(1)

    if (report["STATUS"] == "WARNING").any():
        print("\nValidation warnings require review.")
        raise SystemExit(2)

    print("\nAll game coverage checks passed.")
    print("Note: Player eligibility completeness is not yet certified.")


if __name__ == "__main__":
    main()

