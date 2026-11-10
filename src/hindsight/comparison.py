
"""Compare historical optimizer lineups against actual DraftKings results.

Run:
    python -m src.hindsight.comparison --date 2026-11-06
    python -m src.hindsight.comparison --date 2026-11-06 --save
"""

import argparse
from pathlib import Path

import pandas as pd

from src.hindsight.actuals import prepare_actuals
from src.hindsight.validation import validate_slate


LINEUP_DIR = Path("data/dk_lineups")
HINDSIGHT_DIR = Path("data/dk_hindsight")
STRATEGIES = ("cash", "balanced", "gpp")


def compare_lineups(
    date,
    lineup_dir=LINEUP_DIR,
    hindsight_dir=HINDSIGHT_DIR,
):
    """Score archived optimizer lineups using actual DK points."""

    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    date_tag = date.replace("-", "_")

    validation = validate_slate(date)

    if not (validation["STATUS"] == "PASS").all():
        raise ValueError(
            f"Slate validation did not pass for {date}."
        )

    actuals, _ = prepare_actuals(date)

    actual_lookup = actuals[
        ["DK_ID", "DK_POINTS"]
    ].copy()

    status_file = (
        Path(hindsight_dir) / "player_status_overrides.csv"
    )
    
    if status_file.exists():
        overrides = pd.read_csv(status_file)
    
        overrides = overrides.loc[
            overrides["GAME_DATE"].eq(date)
            & overrides["RESULT_STATUS"].eq("CONFIRMED_DNP"),
            ["DK_ID"]
        ].drop_duplicates()
    
        actual_lookup = actual_lookup.merge(
            overrides.assign(CONFIRMED_DNP=True),
            on="DK_ID",
            how="outer",
            validate="one_to_one",
        )
    
        actual_lookup.loc[
            actual_lookup["CONFIRMED_DNP"].eq(True)
            & actual_lookup["DK_POINTS"].isna(),
            "DK_POINTS"
        ] = 0.0

    perfect_file = (
        Path(hindsight_dir)
        / f"perfect_summary_{date_tag}_Main.csv"
    )

    if not perfect_file.exists():
        raise FileNotFoundError(
            f"Missing hindsight summary: {perfect_file}"
        )

    perfect_summary = pd.read_csv(perfect_file)
    perfect_score = float(
        perfect_summary.iloc[0]["actual_dk_points"]
    )

    summaries = []
    details = []

    for strategy in STRATEGIES:
        lineup_file = (
            Path(lineup_dir)
            / f"lineups_{date_tag}_Main_{strategy}_details.csv"
        )

        if not lineup_file.exists():
            print(f"Skipping missing strategy file: {lineup_file}")
            continue

        lineup = pd.read_csv(lineup_file)

        lineup = lineup.merge(
            actual_lookup,
            on="DK_ID",
            how="left",
            validate="many_to_one",
        )

        lineup["RESULT_STATUS"] = "UNKNOWN"

        lineup.loc[
            lineup["DK_POINTS"].notna(), "RESULT_STATUS"
        ] = "PLAYED"

        if "CONFIRMED_DNP" in lineup.columns:
            lineup.loc[
                lineup["CONFIRMED_DNP"].eq(True)
                & lineup["DK_POINTS"].eq(0),
                "RESULT_STATUS"
            ] = "CONFIRMED_DNP"

        lineup["STRATEGY"] = strategy
        lineup["PROJECTION_ERROR"] = (
            lineup["PROJECTED_DK_POINTS"]
            - lineup["DK_POINTS"]
        )

        details.append(lineup)

        for lineup_number, group in lineup.groupby("LINEUP_NUMBER"):
            missing = int(group["DK_POINTS"].isna().sum())
            complete = missing == 0 and len(group) == 8

            dnp_mask = group["RESULT_STATUS"].eq("CONFIRMED_DNP")
            dnp_count = int(dnp_mask.sum())
            dnp_projected_points = float(
                group.loc[dnp_mask, "PROJECTED_DK_POINTS"].sum()
            )

            projected = group["PROJECTED_DK_POINTS"].sum()

            actual = (
                group["DK_POINTS"].sum()
                if complete
                else float("nan")
            )

            summaries.append({
                "DATE": date,
                "STRATEGY": strategy,
                "LINEUP_NUMBER": lineup_number,
                "PLAYERS": len(group),
                "MISSING_RESULTS": missing,
                "DNP_COUNT": dnp_count,
                "DNP_PROJECTED_POINTS": round(dnp_projected_points, 2),
                "COMPLETE": complete,
                "SALARY": group["SALARY"].sum(),
                "PROJECTED_POINTS": projected,
                "ACTUAL_POINTS": actual,
                "PROJECTION_ERROR": (
                    projected - actual if complete else float("nan")
                ),
                "PERFECT_POINTS": perfect_score,
                "POINTS_BELOW_PERFECT": (
                    perfect_score - actual
                    if complete else float("nan")
                ),
                "PCT_OF_PERFECT": (
                    100 * actual / perfect_score
                    if complete and perfect_score
                    else float("nan")
                ),
            })

    if not summaries:
        raise FileNotFoundError(
            f"No saved strategy lineups found for {date}."
        )

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)

    return summary_df, detail_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare archived NBA optimizer lineups."
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    summary, details = compare_lineups(args.date)

    print(f"\nLINEUP COMPARISON: {args.date}")
    print(summary.to_string(index=False, float_format="%.2f"))

    incomplete = summary.loc[~summary["COMPLETE"]]
    if not incomplete.empty:
        print("\nWARNING: Some lineups have missing player results.")
        print(
            incomplete[
                ["STRATEGY", "LINEUP_NUMBER", "MISSING_RESULTS"]
            ].to_string(index=False)
        )

    if args.save:
        output_dir = HINDSIGHT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        date_tag = args.date.replace("-", "_")

        summary_file = (
            output_dir / f"comparison_{date_tag}_Main_summary.csv"
        )
        detail_file = (
            output_dir / f"comparison_{date_tag}_Main_details.csv"
        )

        summary.to_csv(summary_file, index=False)
        details.to_csv(detail_file, index=False)

        print(f"\nSaved summary: {summary_file}")
        print(f"Saved details: {detail_file}")


if __name__ == "__main__":
    main()
