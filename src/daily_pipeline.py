
# git pull
# python -m src.daily_pipeline --date 2026-11-08


"""Run the complete NBA DraftKings daily production pipeline."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def run_stage(name, command):
    """Execute one pipeline stage and stop on failure."""

    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE: {name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    start = time.time()

    subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        check=True,
    )

    elapsed = time.time() - start
    print(f"\nCompleted: {name} ({elapsed:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the NBA DraftKings daily pipeline."
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    from datetime import datetime

    slate_date = datetime.strptime(
        args.date, "%Y-%m-%d"
    ).date()

    date_tag = slate_date.strftime("%Y_%m_%d")

    salary_file = (
        DATA_DIR
        / "dk_salaries"
        / f"DKSalaries_{date_tag}_Main.csv"
    )

    projection_file = (
        DATA_DIR
        / "dk_projections"
        / f"projections_{date_tag}_Main.pkl"
    )

    if not salary_file.is_file():
        parser.error(f"Salary file not found: {salary_file}")

    print(f"NBA DRAFTKINGS PIPELINE: {slate_date}")
    print(f"Salary file: {salary_file.name}")

    stages = [
        (
            "Update NBA game data",
            ["-m", "src.update_nba_data"],
        ),
        (
            "Refresh DraftKings player mappings",
            ["-m", "src.match_players"],
        ),
        (
            "Rebuild historical features",
            [
                "-m", "src.dk_features",
                "--season", "2026-27",
            ],
        ),
        (
            "Train production LightGBM model",
            [
                "-m", "src.modeling.train_production",
                "--cutoff", str(slate_date),
            ],
        ),
        (
            "Generate slate projections",
            [
                "-m", "src.pregame",
                "--date", str(slate_date),
            ],
        ),
        (
            "Optimize DraftKings lineups",
            [
                "-m", "src.optimizer.run",
                "--projections", str(projection_file),
            ],
        ),
    ]

    pipeline_start = time.time()

    for name, command in stages:
        run_stage(name, command)

    print(f"\n{'=' * 60}")
    print("DAILY PIPELINE COMPLETE")
    print(f"Total runtime: {time.time() - pipeline_start:.1f}s")
    print(f"Projections: {projection_file}")
    print("Lineups: data/dk_lineups/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
