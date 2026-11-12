
import argparse
from pathlib import Path

import pandas as pd

from src.optimizer.classic import optimize_classic

# Run optimizer for a specific slate date:
# python -m src.optimizer.run --projections data/dk_projections/projections_2026_11_10_Main.pkl

# ============================================================
# LINEUP EXPERIMENT SETTINGS
# ============================================================

SLATE_DATE = "2026_11_12"

PROJECTIONS_FILE = (
    f"data/dk_projections/projections_{SLATE_DATE}_Main.pkl"
)

NUM_LINEUPS = 3
MAX_PLAYER_OVERLAP = 5
MIN_PLAYER_SALARY = 3500
MIN_PLAYER_MINUTES = 22

EXCLUDE_QUESTIONABLE = True

EXCLUDED_PLAYERS = [
    "Ryan Dunn",
    #"Anthony Davis",
    #"Dalton Knecht",
    # "Kyrie Irving",
    #"Paul George",
    #"Quenton Jackson",
    #"Obi Toppin"
]


LOCKED_PLAYERS = [
   # "Dyson Daniels",
    #"Luka Doncic",
    #"Victor Wembanyama",
    #"Yaxel Lendeborg",
    # "Kyrie Irving",
    # "Myles Turner",
]

LINEUP_STRATEGY = 2
# 1 - cash: prioritize low-variance players for safer lineups
# 2 - balanced: mix of low- and high-variance players
# 3 - gpp: prioritize high-variance players for upside potential

STRATEGY_WEIGHTS = {
    1: -0.5,
    2: 0.0,
    3: 0.5,
}

STRATEGY_NAMES = {
    1: "cash",
    2: "balanced",
    3: "gpp",
}

# ============================================================




DEFAULT_PROJECTIONS = Path(PROJECTIONS_FILE)

DISPLAY_COLUMNS = [
    "ROSTER_SLOT",
    "DK_NAME",
    "TEAM",
    "OPPONENT",
    "SALARY",
    "DK_AVG_POINTS",
    "PROJECTED_DK_POINTS",
    "PROJECTION_VS_DK_AVG",
    "VALUE_PER_1000",
    "STATUS",
]


def main():
    parser = argparse.ArgumentParser(
        description="Optimize a saved DraftKings NBA Classic slate."
    )
    parser.add_argument(
        "--projections",
        type=Path,
        default=DEFAULT_PROJECTIONS,
    )
    parser.add_argument(
        "--exclude-questionable",
        action="store_true",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="PLAYER",
        help="Exclude players by exact DK_NAME.",
    )
    
    parser.add_argument(
    "--min-salary",
    type=int,
    default=MIN_PLAYER_SALARY,
    help="Minimum player salary (default: %(default)s).",
    )

    parser.add_argument(
        "--min-minutes",
        type=float,
        default=MIN_PLAYER_MINUTES,
        help="Minimum average minutes over last 5 games (default: %(default)s).",
    )

    args = parser.parse_args()

    if args.min_salary < 0:
        parser.error("--min-salary cannot be negative.")

    if not 0 <= args.min_minutes <= 48:
        parser.error("--min-minutes must be between 0 and 48.")

    slate = pd.read_pickle(args.projections)

        # Display questionable players for pregame injury research.
    injury_review = (
        slate.loc[
            slate["IS_QUESTIONABLE"],
            ["DK_NAME", "TEAM", "SALARY", "PROJECTED_DK_POINTS", "STATUS"],
        ]
        .sort_values("PROJECTED_DK_POINTS", ascending=False)
        .copy()
    )

    print("\nQUESTIONABLE PLAYERS TO REVIEW")
    if injury_review.empty:
        print("No questionable players.")
    else:
        print(
            injury_review.to_string(
                index=False,
                formatters={
                    "PROJECTED_DK_POINTS": "{:.2f}".format,
                },
            )
        )

    if EXCLUDE_QUESTIONABLE or args.exclude_questionable:
        slate["OPTIMIZER_ELIGIBLE"] = (
            slate["OPTIMIZER_ELIGIBLE"]
            & ~slate["IS_QUESTIONABLE"]
        )

    if EXCLUDED_PLAYERS or args.exclude:
        excluded_players = set(EXCLUDED_PLAYERS) | set(args.exclude)

        if excluded_players:
            unknown = excluded_players - set(slate["DK_NAME"])

            if unknown:
                parser.error(
                    f"Unknown player names: {sorted(unknown)}"
                )

            slate["OPTIMIZER_ELIGIBLE"] = (
                slate["OPTIMIZER_ELIGIBLE"]
                & ~slate["DK_NAME"].isin(excluded_players)
            )

    print(f"Minimum player salary: ${args.min_salary:,}")

    print(f"Minimum average minutes: {args.min_minutes:g}")

    if args.min_minutes > 0:
        minutes = pd.to_numeric(
            slate["MIN_AVG_LAST_5"],
            errors="coerce",
        )

        eligible = slate["OPTIMIZER_ELIGIBLE"].eq(True)
        salary_ok = pd.to_numeric(
            slate["SALARY"], errors="coerce"
        ).ge(args.min_salary)

        candidate_pool = eligible & salary_ok

        excluded_by_minutes = (
            candidate_pool & ~minutes.ge(args.min_minutes)
        )

        print(
            f"Players excluded by minutes cutoff: "
            f"{excluded_by_minutes.sum()}"
        )
        print(
            f"Eligible players remaining: "
            f"{(candidate_pool & ~excluded_by_minutes).sum()}"
        )




    # Calculate strategy-adjusted optimization scores
    if LINEUP_STRATEGY not in STRATEGY_WEIGHTS:
        raise ValueError(
            f"Unknown lineup strategy: {LINEUP_STRATEGY}"
        )

    slate["OPTIMIZATION_SCORE"] = (
        slate["PROJECTED_DK_POINTS"]
        + STRATEGY_WEIGHTS[LINEUP_STRATEGY]
        * slate["DK_STD_LAST_10"].fillna(0)
    )

    print(f"\nLineup strategy: {STRATEGY_NAMES[LINEUP_STRATEGY]}")

    lineup_summaries = []
    lineup_details = []
    previous_lineups = []

    for lineup_number in range(1, NUM_LINEUPS + 1):
        try:
            result = optimize_classic(
                slate,
                locked_players=LOCKED_PLAYERS,
                previous_lineups=previous_lineups,
                max_player_overlap=MAX_PLAYER_OVERLAP,
                min_salary=args.min_salary,
                min_minutes=args.min_minutes,
            )
        except RuntimeError as exc:
            print(
                f"\nCould not generate lineup {lineup_number}: {exc}"
            )
            break

        lineup = result["lineup"]

                # Historical volatility measurements
        player_std = pd.to_numeric(
            lineup["DK_STD_LAST_10"],
            errors="coerce",
        )

        avg_player_volatility = player_std.mean()

        # Assumes independent player outcomes
        lineup_volatility = (
            (player_std ** 2).sum() ** 0.5
        )

        export_lineup = lineup.copy()
        export_lineup.insert(0, "LINEUP_NUMBER", lineup_number)
        lineup_details.append(export_lineup)

        print(f"\nLINEUP {lineup_number}")
        print(
            lineup[DISPLAY_COLUMNS].to_string(
                index=False,
                formatters={
                    "DK_AVG_POINTS": "{:.2f}".format,
                    "PROJECTED_DK_POINTS": "{:.2f}".format,
                    "PROJECTION_VS_DK_AVG": "{:+.2f}".format,
                    "VALUE_PER_1000": "{:.2f}".format,
                },
            )
        )
        print(f"Total salary: ${result['total_salary']:,}")
        print(f"Remaining salary: ${result['remaining_salary']:,}")
        print(
            f"Projected DK points: "
            f"{result['projected_points']:.2f}"
        )

        previous_lineups.append(
            set(lineup["DK_ID"])
        )
        
        lineup_summaries.append({
            "Lineup": lineup_number,
            "Strategy": STRATEGY_NAMES[LINEUP_STRATEGY],
            "Min Salary": args.min_salary,
            "Salary": result["total_salary"],
            "Projected Points": round(result["projected_points"], 2),
            "Avg Player STD": round(avg_player_volatility, 2),
            "Lineup STD Proxy": round(lineup_volatility, 2),
            "Players": ", ".join(lineup["DK_NAME"]),
            "Min Minutes": args.min_minutes,
            "Avg Lineup Minutes": round(
                lineup["MIN_AVG_LAST_5"].mean(), 2
            ),
        })
    
    if lineup_summaries:
        summary = pd.DataFrame(lineup_summaries)

        print(
            summary[
                [
                    "Lineup",
                    "Strategy",
                    "Salary",
                    "Projected Points",
                    "Avg Player STD",
                    "Lineup STD Proxy",
                ]
            ].to_string(index=False)
        )

    if lineup_details:
        output_dir = Path("data/dk_lineups")
        output_dir.mkdir(parents=True, exist_ok=True)

        slate_date = pd.to_datetime(
            slate["GAME_DATE"].iloc[0]
        ).strftime("%Y_%m_%d")

        strategy_name = (
            f"{STRATEGY_NAMES[LINEUP_STRATEGY]}"
            f"_min{args.min_salary}"
            f"_minutes{args.min_minutes:g}"
        )

        details_file = (
            output_dir
            / f"lineups_{slate_date}_Main_{strategy_name}_details.csv"
        )
        summary_file = (
            output_dir
            / f"lineups_{slate_date}_Main_{strategy_name}_summary.csv"
        )
        details = pd.concat(lineup_details, ignore_index=True)

        details_columns = [
            "LINEUP_NUMBER",
            "ROSTER_SLOT",
            "DK_ID",
            "DK_NAME",
            "TEAM",
            "OPPONENT",
            "SALARY",
            "DK_AVG_POINTS",
            "MIN_AVG_LAST_5",
            "PROJECTED_DK_POINTS",
            "PROJECTION_VS_DK_AVG",
            "VALUE_PER_1000",
            "STATUS",
        ]

        details[details_columns].to_csv(details_file, index=False)
        summary.to_csv(summary_file, index=False)

        print(f"\nSaved lineup details: {details_file}")
        print(f"Saved lineup summary: {summary_file}")

if __name__ == "__main__":
    main()
