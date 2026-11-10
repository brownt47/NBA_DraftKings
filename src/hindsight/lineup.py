
"""Generate a hindsight-optimal DraftKings NBA Classic lineup.

Run:
    python -m src.hindsight.lineup --date 2026-11-08
    python -m src.hindsight.lineup --date 2026-11-08 --save
"""

import argparse
from pathlib import Path

import pandas as pd

from src.hindsight.actuals import prepare_actuals
from src.optimizer.classic import optimize_classic
from src.hindsight.validation import validate_slate


OUTPUT_DIR = Path("data/dk_hindsight")

DISPLAY_COLUMNS = [
    "ROSTER_SLOT",
    "DK_NAME",
    "TEAM_DK",
    "SALARY",
    "DK_POINTS",
]


def build_perfect_lineup(date, allow_warnings=False):
    """Validate the slate, then optimize actual DK points."""
    validation = validate_slate(date)

    if (validation["STATUS"] == "FAIL").any():
        raise ValueError(
            f"Slate validation failed for {date}:\n"
            f"{validation.to_string(index=False)}"
        )

    if (validation["STATUS"] == "WARNING").any() and not allow_warnings:
        raise ValueError(
            f"Slate validation warnings for {date}:\n"
            f"{validation.to_string(index=False)}\n"
            "Use --allow-warnings to proceed."
        )

    players, report = prepare_actuals(date)

    result = optimize_classic(
        players,
        min_salary=0,
    )

    report["validation_status"] = (
        "WARNING"
        if (validation["STATUS"] == "WARNING").any()
        else "PASS"
    )
    report["validated_games"] = len(validation)

    return result, report


def save_lineup(result, report, output_dir=OUTPUT_DIR):
    """Save hindsight lineup and summary independently."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date = report["date"].replace("-", "_")

    lineup_file = output_dir / f"perfect_lineup_{date}_Main.csv"
    summary_file = output_dir / f"perfect_summary_{date}_Main.csv"

    lineup = result["lineup"].copy()
    lineup["ROSTER_SLOT"] = lineup["ROSTER_SLOT"].astype(str)

    lineup[DISPLAY_COLUMNS].to_csv(lineup_file, index=False)

    summary = pd.DataFrame([{
        **report,
        "total_salary": result["total_salary"],
        "remaining_salary": result["remaining_salary"],
        "actual_dk_points": result["optimization_score"],
        "certified_complete": False,
    }])

    summary.to_csv(summary_file, index=False)

    return lineup_file, summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Build a hindsight-optimal NBA Classic lineup."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Historical slate date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save lineup and summary CSV files.",
    )

    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Allow optimization when validation has warnings.",
    )

    args = parser.parse_args()

    result, report = build_perfect_lineup(
        args.date,
        allow_warnings=args.allow_warnings,
    )

    lineup = result["lineup"]

    print(f"\nHINDSIGHT OPTIMAL LINEUP: {report['date']}")
    print(
        lineup[DISPLAY_COLUMNS].to_string(index=False)
    )

    print(f"\nTotal salary: ${result['total_salary']:,}")
    print(f"Actual DK points: {result['optimization_score']:.2f}")
    print(f"Remaining salary: ${result['remaining_salary']:,}")

    print("\nSLATE VALIDATION")
    print(f"Status: {report['validation_status']}")
    print(f"Validated games: {report['validated_games']}")

    print("\nDATA COVERAGE")
    print(f"Salary players: {report['salary_players']}")
    print(f"Mapped players: {report['mapped_players']}")
    print(f"Players with results: {report['players_with_results']}")
    print(f"Players without results: {report['players_without_results']}")
    print(f"Games with results: {report['games_with_results']}")

    print(
        "\nNote: This is the optimum among players with recorded "
        "results. Full slate completeness has not yet been certified."
    )

    if args.save:
        lineup_file, summary_file = save_lineup(result, report)
        print(f"\nSaved lineup: {lineup_file}")
        print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
