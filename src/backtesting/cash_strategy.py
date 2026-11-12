"""Backtest pregame NBA Classic risk penalties against historical DK results.

Run:
    python -m src.backtesting.cash_strategy --date 2026-11-08
    python -m src.backtesting.cash_strategy --date 2026-11-08 --save
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.hindsight.actuals import prepare_actuals
from src.hindsight.validation import validate_slate
from src.optimizer.classic import optimize_classic

PROJECTION_DIR = Path("data/dk_projections")
HINDSIGHT_DIR = Path("data/dk_hindsight")
WEIGHTS = (0.0, -0.1, -0.25, -0.5, -0.75, -1.0)


def backtest_cash_strategies(date, min_salary=4000, exclude_questionable=True):
    """Optimize archived pregame projections, then score against actuals."""
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    tag = date.replace("-", "_")
    projection_file = PROJECTION_DIR / f"projections_{tag}_Main.pkl"
    if not projection_file.is_file():
        raise FileNotFoundError(projection_file)

    validation = validate_slate(date)
    if (validation["STATUS"] == "FAIL").any():
        raise ValueError(f"Incomplete game coverage for {date}:\n{validation}")
    if (validation["STATUS"] == "WARNING").any():
        print("WARNING: Slate validation needs review (possibly overtime).")
        print(validation.loc[validation["STATUS"] == "WARNING"].to_string(index=False))

    slate = pd.read_pickle(projection_file).copy()
    if "GAME_DATE" in slate.columns:
        dates = pd.to_datetime(slate["GAME_DATE"]).dt.strftime("%Y-%m-%d")
        if not dates.eq(date).all():
            raise ValueError("Projection file contains rows from a different date.")
    if exclude_questionable:
        if "IS_QUESTIONABLE" not in slate.columns:
            raise ValueError("Missing IS_QUESTIONABLE in projection archive.")
        slate["OPTIMIZER_ELIGIBLE"] = (
            slate["OPTIMIZER_ELIGIBLE"].eq(True)
            & ~slate["IS_QUESTIONABLE"].fillna(False).astype(bool)
        )

    actuals, coverage = prepare_actuals(date)
    actual_lookup = actuals[["DK_ID", "DK_POINTS"]].copy()
    actual_lookup["DK_ID"] = pd.to_numeric(actual_lookup["DK_ID"], errors="raise")
    actual_lookup["DK_POINTS"] = pd.to_numeric(actual_lookup["DK_POINTS"], errors="raise")
    if actual_lookup["DK_ID"].duplicated().any():
        raise ValueError("Duplicate DK IDs in actual results.")

    status_file = HINDSIGHT_DIR / "player_status_overrides.csv"
    if status_file.is_file():
        overrides = pd.read_csv(status_file)
        confirmed = overrides.loc[
            overrides["GAME_DATE"].eq(date)
            & overrides["RESULT_STATUS"].eq("CONFIRMED_DNP"), "DK_ID"
        ].dropna()
        confirmed = set(pd.to_numeric(confirmed, errors="raise"))
    else:
        confirmed = set()

    summaries, details = [], []
    for weight in WEIGHTS:
        candidate = slate.copy()
        std = pd.to_numeric(candidate["DK_STD_LAST_10"], errors="coerce")
        # Replicate production's missing-STD behavior for a fair baseline.
        candidate["OPTIMIZATION_SCORE"] = (
            pd.to_numeric(candidate["PROJECTED_DK_POINTS"], errors="coerce")
            + weight * std.fillna(0)
        )
        try:
            result = optimize_classic(candidate, min_salary=min_salary)
        except RuntimeError as exc:
            summaries.append({"DATE": date, "WEIGHT": weight, "STATUS": str(exc),
                              "COMPLETE": False})
            continue

        lineup = result["lineup"].copy()
        lineup["DK_ID"] = pd.to_numeric(lineup["DK_ID"], errors="raise")
        lineup = lineup.merge(actual_lookup, on="DK_ID", how="left", validate="one_to_one")
        dnp = lineup["DK_ID"].isin(confirmed)
        lineup.loc[dnp & lineup["DK_POINTS"].isna(), "DK_POINTS"] = 0.0
        lineup["RESULT_STATUS"] = np.where(
            dnp, "CONFIRMED_DNP", np.where(lineup["DK_POINTS"].notna(), "PLAYED", "UNKNOWN")
        )
        missing = int(lineup["DK_POINTS"].isna().sum())
        complete = len(lineup) == 8 and missing == 0
        player_std = pd.to_numeric(lineup["DK_STD_LAST_10"], errors="coerce")
        summaries.append({
            "DATE": date, "WEIGHT": weight, "STATUS": "OK" if complete else "MISSING_ACTUALS",
            "COMPLETE": complete, "SALARY": result["total_salary"],
            "PROJECTED_POINTS": round(result["projected_points"], 2),
            "ACTUAL_POINTS": round(float(lineup["DK_POINTS"].sum()), 2) if complete else np.nan,
            "PROJECTION_ERROR": round(float(result["projected_points"] - lineup["DK_POINTS"].sum()), 2) if complete else np.nan,
            "MISSING_RESULTS": missing, "DNP_COUNT": int(dnp.sum()),
            "LINEUP_STD_PROXY": round(float(np.sqrt((player_std ** 2).sum())), 2),
            "MISSING_STD": int(player_std.isna().sum()),
            "PLAYERS": ", ".join(lineup["DK_NAME"].astype(str)),
        })
        lineup.insert(0, "WEIGHT", weight)
        lineup.insert(0, "DATE", date)
        details.append(lineup)

    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True) if details else pd.DataFrame(), coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--min-salary", type=int, default=4000)
    parser.add_argument("--include-questionable", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    summary, details, coverage = backtest_cash_strategies(
        args.date, min_salary=args.min_salary,
        exclude_questionable=not args.include_questionable,
    )
    print("\nCASH STRATEGY BACKTEST")
    print(summary.drop(columns=["PLAYERS"], errors="ignore").to_string(index=False))
    print("\nACTUALS COVERAGE:", coverage)
    if args.save:
        output = Path("data/dk_backtests")
        output.mkdir(parents=True, exist_ok=True)
        tag = args.date.replace("-", "_")
        summary.to_csv(output / f"cash_strategy_{tag}_summary.csv", index=False)
        details.to_csv(output / f"cash_strategy_{tag}_details.csv", index=False)
        print("Saved backtest results in", output)


if __name__ == "__main__":
    main()
