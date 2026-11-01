
# ---------------------------------------------------------
# NBA DATA UPDATE CONTROLLER
# Run player and team scrapers in sequence.
# ---------------------------------------------------------

import subprocess
import sys


# ---------------------------------------------------------
# RUN SCRAPER
# Execute a Python module and stop if it fails.
# ---------------------------------------------------------

def run_scraper(module_name):

    print(f"\nRunning {module_name}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 40)
    print("NBA DATA UPDATE STARTED")
    print("=" * 40)

    run_scraper("src.nba_scraper")
    run_scraper("src.nba_team_scraper")

    print("\n" + "=" * 40)
    print("NBA DATA UPDATE COMPLETE")
    print("=" * 40)


if __name__ == "__main__":
    main()
