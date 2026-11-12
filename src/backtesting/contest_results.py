
"""Compare saved cash-strategy backtests with DraftKings standings.

Example:
    python -m src.backtesting.contest_results --date 2026-11-09

Cash lines are estimates unless actual paid places are supplied.
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd


BACKTEST_DIR = Path("data/dk_backtests")
CONTEST_DIR = Path("data/dk_contests")
OUTPUT_DIR = Path("data/dk_backtests")

FILENAME_PATTERN = re.compile(
    r"contest-standings-(\d+)_(\d{4})_(\d{2})_(\d{2})"
)


def read_contest(path, payout_fraction=0.45, paid_places=None):
    """Read standings and estimate the double-up payout cutoff."""
    path = Path(path)
    match = FILENAME_PATTERN.search(path.name)

    if not match:
        raise ValueError(f"Unrecognized contest filename: {path.name}")

    contest_id, year, month, day = match.groups()
    date = f"{year}-{month}-{day}"

    df = pd.read_csv(path, low_memory=False)

    required = {"EntryId", "Rank", "Points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    standings = df.dropna(
        subset=["EntryId", "Rank", "Points"]
    ).copy()

    standings["Points"] = pd.to_numeric(
        standings["Points"], errors="raise"
    )
    standings["Rank"] = pd.to_numeric(
        standings["Rank"], errors="raise"
    )

    if standings["EntryId"].duplicated().any():
        raise ValueError(f"Duplicate entries in {path.name}")

    standings = standings.sort_values(
        ["Points", "Rank"], ascending=[False, True]
    )

    n = len(standings)
    if n == 0:
        raise ValueError(f"No standings found in {path.name}")

    if paid_places is None:
        if not 0 < payout_fraction <= 1:
            raise ValueError("Payout fraction must be between 0 and 1")
        places = max(1, math.floor(n * payout_fraction))
        cutoff_type = "ESTIMATED"
    else:
        places = int(paid_places)
        if not 1 <= places <= n:
            raise ValueError("Invalid paid places")
        cutoff_type = "PAID_PLACES_SUPPLIED"

    cutoff = float(standings.iloc[places - 1]["Points"])

    return {
        "DATE": date,
        "CONTEST_ID": contest_id,
        "ENTRIES": n,
        "PAID_PLACES": places,
        "CUTOFF": cutoff,
        "CUTOFF_TYPE": cutoff_type,
        "TIED_AT_CUTOFF": int(
            (standings["Points"] == cutoff).sum()
        ),
    }


def compare(date, payout_fraction=0.45, paid_places=None):
    """Compare saved optimizer results against contest cutoffs."""
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    tag = date.replace("-", "_")

    summary_path = (
        BACKTEST_DIR / f"cash_strategy_{tag}_summary.csv"
    )

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    strategies = pd.read_csv(summary_path)
    strategies = strategies.loc[
        strategies["COMPLETE"].astype(str).str.lower().eq("true")
    ].copy()

    if strategies.empty:
        raise ValueError("No complete backtest lineups available")

    contests = []

    for path in sorted(CONTEST_DIR.glob(f"contest-standings-*_{tag}*.csv")):
        contest = read_contest(
            path,
            payout_fraction=payout_fraction,
            paid_places=paid_places,
        )
        if contest["DATE"] == date:
            contests.append(contest)

    if not contests:
        raise FileNotFoundError(
            f"No contest standings for {date} in {CONTEST_DIR}"
        )

    results = []

    for contest in contests:
        for _, strategy in strategies.iterrows():
            score = float(strategy["ACTUAL_POINTS"])
            cutoff = contest["CUTOFF"]

            if score > cutoff:
                status = "ABOVE_CUTOFF"
            elif score < cutoff:
                status = "BELOW_CUTOFF"
            else:
                status = "TIED_AT_CUTOFF"

            results.append({
                **contest,
                "WEIGHT": strategy["WEIGHT"],
                "LINEUP_SCORE": score,
                "MARGIN": round(score - cutoff, 2),
                "RESULT": status,
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--payout-fraction", type=float, default=0.45)
    parser.add_argument("--paid-places", type=int)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results = compare(
        args.date,
        payout_fraction=args.payout_fraction,
        paid_places=args.paid_places,
    )

    print("\nDOUBLE-UP CONTEST COMPARISON")
    print(results.to_string(index=False))

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = args.date.replace("-", "_")

        # Save detailed strategy comparisons.
        path = OUTPUT_DIR / f"contest_comparison_{tag}.csv"
        results.to_csv(path, index=False)
        print(f"\nSaved: {path}")

        # Update historical contest cash lines.
        history_path = OUTPUT_DIR / "contest_cash_lines.csv"

        columns = [
            "DATE", "CONTEST_ID", "ENTRIES",
            "PAID_PLACES", "CUTOFF", "CUTOFF_TYPE",
            "TIED_AT_CUTOFF",
        ]

        new_records = results[columns].drop_duplicates()

        if history_path.exists():
            history = pd.read_csv(
                history_path,
                dtype={"CONTEST_ID": str},
            )
            history = pd.concat(
                [history, new_records], ignore_index=True
            )
        else:
            history = new_records.copy()

    # Keep verified records over estimates, then deduplicate.
    history["CONTEST_ID"] = history["CONTEST_ID"].astype(str)
    history["_VERIFIED"] = history["CUTOFF_TYPE"].eq(
        "PAID_PLACES_SUPPLIED"
    )

    history = (
        history.sort_values("_VERIFIED")
        .drop_duplicates(subset=["CONTEST_ID"], keep="last")
        .drop(columns="_VERIFIED")
        .sort_values(["DATE", "CONTEST_ID"])
    )

    history.to_csv(history_path, index=False)
    print(f"Updated: {history_path} ({len(history)} contests)")


if __name__ == "__main__":
    main()
