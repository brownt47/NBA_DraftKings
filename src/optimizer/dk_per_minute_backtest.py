
"""Historical DraftKings points-per-minute eligibility experiment."""

from pathlib import Path
import re

import pandas as pd

from src.dk_features import load_player_games
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BACKTEST = DATA / "dk_backtest"

THRESHOLDS = [0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
MIN_MINUTES = 22
MIN_SALARY = 3500


def main():
    projection_files = sorted(
        (BACKTEST / "projections").glob("projections_*.pkl")
    )

    if not projection_files:
        raise FileNotFoundError("No historical projections found.")

    players = load_player_games()
    results = []

    for path in projection_files:
        match = re.search(
            r"projections_(\d{4})_(\d{2})_(\d{2})\.pkl$",
            path.name,
        )

        if not match:
            continue

        date = "-".join(match.groups())

        # Do not evaluate an unfinished slate.
        actual = load_actual_results(players, date)
        if actual.empty:
            print(f"Skipping {date}: no actual results")
            continue

        slate = pd.read_pickle(path)

        print(f"\nProcessing {date}")

        for threshold in THRESHOLDS:
            try:
                result = optimize_classic(
                    slate,
                    min_salary=MIN_SALARY,
                    min_minutes=MIN_MINUTES,
                    min_dk_per_minute=threshold,
                )

                lineup = result["lineup"].copy()
                lineup["PLAYER_ID"] = lineup["PLAYER_ID"].astype(str)

                lineup = lineup.merge(
                    actual,
                    on="PLAYER_ID",
                    how="left",
                    validate="one_to_one",
                )

                missing = int(
                    lineup["ACTUAL_DK_POINTS"].isna().sum()
                )

                actual_points = (
                    float(lineup["ACTUAL_DK_POINTS"].sum())
                    if missing == 0
                    else float("nan")
                )

                status = (
                    "COMPLETE" if missing == 0
                    else "MISSING_RESULTS"
                )

                results.append({
                    "DATE": date,
                    "MIN_MINUTES": MIN_MINUTES,
                    "MIN_DK_PER_MINUTE": threshold,
                    "STATUS": status,
                    "PROJECTED": result["projected_points"],
                    "ACTUAL": actual_points,
                    "MISSING_RESULTS": missing,
                })

                print(
                    f"  DK/min {threshold:.2f}: "
                    f"Projected {result['projected_points']:.2f}, "
                    f"Actual {actual_points:.2f}"
                )

            except (ValueError, RuntimeError) as exc:
                results.append({
                    "DATE": date,
                    "MIN_MINUTES": MIN_MINUTES,
                    "MIN_DK_PER_MINUTE": threshold,
                    "STATUS": "FAILED",
                    "ERROR_MESSAGE": str(exc),
                })

                print(f"  DK/min {threshold:.2f}: FAILED")

    df = pd.DataFrame(results)

    output = BACKTEST / "results" / "dk_per_minute_backtest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    complete = df.loc[df["STATUS"].eq("COMPLETE")].copy()

    if complete.empty:
        print("\nNo completed results.")
        return

    baseline = complete.loc[
        complete["MIN_DK_PER_MINUTE"].eq(0),
        ["DATE", "ACTUAL"],
    ].rename(columns={"ACTUAL": "BASELINE_ACTUAL"})

    complete = complete.merge(
        baseline,
        on="DATE",
        how="left",
        validate="many_to_one",
    )

    complete["IMPROVEMENT"] = (
        complete["ACTUAL"] - complete["BASELINE_ACTUAL"]
    )

    summary = (
        complete.groupby("MIN_DK_PER_MINUTE")
        .agg(
            SLATES=("DATE", "nunique"),
            AVG_ACTUAL=("ACTUAL", "mean"),
            AVG_IMPROVEMENT=("IMPROVEMENT", "mean"),
            MEDIAN_IMPROVEMENT=("IMPROVEMENT", "median"),
            WINS=("IMPROVEMENT", lambda x: (x > 0).sum()),
            LOSSES=("IMPROVEMENT", lambda x: (x < 0).sum()),
            TIES=("IMPROVEMENT", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    print("\nDK/MINUTE HISTORICAL COMPARISON")
    print("=" * 105)
    print(summary.round(2).to_string(index=False))

    summary.to_csv(
        BACKTEST / "results" / "dk_per_minute_summary.csv",
        index=False,
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
