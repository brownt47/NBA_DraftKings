"""Analyze how minimum playing-time thresholds affect optimal DK lineups."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimizer.classic import optimize_classic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--min-salary", type=int, default=3500)
    parser.add_argument("--max-minutes", type=int, default=32)
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args()

    if args.min_salary < 0 or args.max_minutes < 0 or args.step <= 0:
        parser.error("Invalid salary, maximum minutes, or step.")

    slate = pd.read_pickle(args.projections).copy()

    # Match our current optimizer settings.
    slate["OPTIMIZER_ELIGIBLE"] = (
        slate["OPTIMIZER_ELIGIBLE"].eq(True)
        & ~slate["IS_QUESTIONABLE"].eq(True)
    )

    slate["OPTIMIZATION_SCORE"] = slate["PROJECTED_DK_POINTS"]

    results = []
    previous_lineup = None

    for threshold in range(0, args.max_minutes + 1, args.step):
        eligible = (
            slate["OPTIMIZER_ELIGIBLE"]
            & pd.to_numeric(
                slate["SALARY"], errors="coerce"
            ).ge(args.min_salary)
        )

        if threshold > 0:
            eligible &= pd.to_numeric(
                slate["MIN_AVG_LAST_5"], errors="coerce"
            ).ge(threshold)

        pool = slate.loc[eligible]

        record = {
            "MIN_MINUTES": threshold,
            "ELIGIBLE_PLAYERS": len(pool),
            "CHEAP_PLAYERS_UNDER_5000": pool["SALARY"].lt(5000).sum(),
            "PROJECTED_POINTS": np.nan,
            "SALARY": np.nan,
            "LINEUP_CHANGED": np.nan,
            "PLAYERS": "",
        }

        try:
            optimized = optimize_classic(
                slate,
                min_salary=args.min_salary,
                min_minutes=threshold,
            )

            lineup = optimized["lineup"]
            current_lineup = set(lineup["DK_ID"])

            record["PROJECTED_POINTS"] = optimized["projected_points"]
            record["SALARY"] = optimized["total_salary"]
            record["LINEUP_CHANGED"] = (
                previous_lineup is not None
                and current_lineup != previous_lineup
            )
            record["PLAYERS"] = ", ".join(lineup["DK_NAME"])

            previous_lineup = current_lineup

        except (RuntimeError, ValueError) as exc:
            record["ERROR"] = str(exc)

        results.append(record)

    summary = pd.DataFrame(results)

    baseline = summary.loc[
        summary["MIN_MINUTES"].eq(0), "PROJECTED_POINTS"
    ].iloc[0]

    summary["POINTS_LOST"] = baseline - summary["PROJECTED_POINTS"]

    summary["DROP_FROM_PREVIOUS"] = (
        summary["PROJECTED_POINTS"].shift(1)
        - summary["PROJECTED_POINTS"]
    )

    print("\nMINUTES SENSITIVITY ANALYSIS")
    print(
        summary[
            [
                "MIN_MINUTES",
                "ELIGIBLE_PLAYERS",
                "CHEAP_PLAYERS_UNDER_5000",
                "PROJECTED_POINTS",
                "POINTS_LOST",
                "DROP_FROM_PREVIOUS",
                "LINEUP_CHANGED",
            ]
        ].to_string(index=False)
    )

    output_dir = Path("reports/optimizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"minutes_sensitivity_{args.projections.stem}.csv"
    )

    summary.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()