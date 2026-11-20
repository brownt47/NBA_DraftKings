"""Update NBA player, participation, and team data."""

import subprocess
import sys


def run_scraper(module_name, *args):
    print(f"\nRunning {module_name}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name, *args],
        check=True,
    )


def main():
    print("=" * 40)
    print("NBA DATA UPDATE STARTED")
    print("=" * 40)

    run_scraper("src.nba_scraper", "--participation")
    run_scraper("src.nba_team_scraper")

    print("\n" + "=" * 40)
    print("NBA DATA UPDATE COMPLETE")
    print("=" * 40)


if __name__ == "__main__":
    main()
