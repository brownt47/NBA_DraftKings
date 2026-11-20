
from pathlib import Path
import pandas as pd

from src.dk_features import load_player_games
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path("data/dk_backtest")
RELIABILITY = Path(
    "reports/modeling/reliability/reliability_v1_predictions.csv"
)
OUTPUT = Path("reports/modeling/reliability")

MIN_SALARY = 3500
MIN_MINUTES = 23
TEAM_CAP = 3

CONFIGS = {
    "HARD_23": None,
    "RELIABILITY_50": 0.50,
    "RELIABILITY_60": 0.60,
}


def main():
    predictions = pd.read_csv(
        RELIABILITY,
        dtype={"PLAYER_ID": str},
    )
    predictions["GAME_DATE"] = pd.to_datetime(
        predictions["GAME_DATE"]
    ).dt.normalize()

    required = {"PLAYER_ID", "GAME_DATE", "MIN23_FULL"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing reliability columns: {missing}")

    predictions["MIN23_FULL"] = pd.to_numeric(
        predictions["MIN23_FULL"], errors="coerce"
    )

    if predictions.duplicated(["PLAYER_ID", "GAME_DATE"]).any():
        raise ValueError("Duplicate reliability player-date predictions")

    players = load_player_games()
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    players["PLAYER_ID"] = players["PLAYER_ID"].astype(str)

    paths = sorted(
        list((ROOT / "projections").glob("projections_*.pkl"))
        + list(
            (ROOT / "extended" / "projections")
            .glob("projections_*.pkl")
        )
    )

    summaries = []
    details = []

    for path in paths:
        date = pd.Timestamp(
            path.stem.replace("projections_", "").replace("_", "-")
        )

        probs = predictions.loc[
            predictions["GAME_DATE"].eq(date),
            ["PLAYER_ID", "MIN23_FULL"],
        ]

        if probs.empty:
            continue

        print(f"\n{'=' * 65}")
        print(f"SLATE {date.date()}")
        print("=" * 65)

        slate = pd.read_pickle(path).copy()

        if "PLAYER_ID" not in slate.columns:
            raise ValueError(f"{path}: missing PLAYER_ID")

        slate["PLAYER_ID"] = slate["PLAYER_ID"].astype(str)

        if slate["PLAYER_ID"].duplicated().any():
            raise ValueError(f"{path}: duplicate PLAYER_ID")

        slate = slate.merge(
            probs,
            on="PLAYER_ID",
            how="left",
            validate="one_to_one",
        )

        original_eligible = slate["OPTIMIZER_ELIGIBLE"].eq(True)
        minutes = pd.to_numeric(
            slate["MIN_AVG_LAST_5"], errors="coerce"
        )
        hard_floor = minutes.ge(MIN_MINUTES)

        actual = load_actual_results(players, str(date.date()))

        for config, threshold in CONFIGS.items():
            trial = slate.copy()

            if threshold is None:
                exception = pd.Series(
                    False, index=trial.index
                )
            else:
                exception = (
                    minutes.lt(MIN_MINUTES)
                    & trial["MIN23_FULL"].ge(threshold)
                )

            trial["OPTIMIZER_ELIGIBLE"] = (
                original_eligible & (hard_floor | exception)
            )

            try:
                result = optimize_classic(
                    trial,
                    min_salary=MIN_SALARY,
                    min_minutes=0,
                    max_players_per_team=TEAM_CAP,
                )
            except (ValueError, RuntimeError) as exc:
                print(f"{config}: FAILED ({exc})")
                summaries.append({
                    "DATE": str(date.date()),
                    "CONFIG": config,
                    "STATUS": "FAILED",
                    "ERROR": str(exc),
                })
                continue

            lineup = result["lineup"].copy()
            lineup["PLAYER_ID"] = lineup["PLAYER_ID"].astype(str)

            lineup = lineup.merge(
                actual,
                on="PLAYER_ID",
                how="left",
                validate="one_to_one",
            )

            lineup["EXCEPTION_SELECTED"] = (
                pd.to_numeric(
                    lineup["MIN_AVG_LAST_5"],
                    errors="coerce",
                ).lt(MIN_MINUTES)
            )

            n_exceptions = int(
                lineup["EXCEPTION_SELECTED"].sum()
            )
            n_missing = int(
                lineup["ACTUAL_DK_POINTS"].isna().sum()
            )

            actual_points = (
                float(lineup["ACTUAL_DK_POINTS"].sum())
                if n_missing == 0
                else float("nan")
            )

            print(
                f"{config:17s} "
                f"Projected={result['projected_points']:7.2f} "
                f"Actual={actual_points:7.2f} "
                f"Exceptions={n_exceptions} "
                f"Missing={n_missing}"
            )

            summaries.append({
                "DATE": str(date.date()),
                "CONFIG": config,
                "STATUS": (
                    "COMPLETE" if n_missing == 0
                    else "MISSING_RESULTS"
                ),
                "PROJECTED": result["projected_points"],
                "ACTUAL": actual_points,
                "SALARY": result["total_salary"],
                "EXCEPTIONS_SELECTED": n_exceptions,
                "MISSING_RESULTS": n_missing,
                "ELIGIBLE_PLAYERS": int(
                    trial["OPTIMIZER_ELIGIBLE"].sum()
                ),
            })

            lineup["DATE"] = str(date.date())
            lineup["CONFIG"] = config
            details.append(lineup)

    summary = pd.DataFrame(summaries)

    if summary.empty:
        raise RuntimeError("No matching projection dates found")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT / "reliability_lineup_backtest.csv",
        index=False,
    )

    pd.concat(details, ignore_index=True).to_csv(
        OUTPUT / "reliability_lineup_details.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("LINEUP COMPARISON")
    print("=" * 80)

    scores = summary.pivot(
        index="DATE",
        columns="CONFIG",
        values="ACTUAL",
    )

    if "HARD_23" in scores.columns:
        baseline = scores["HARD_23"]
        for config in ["RELIABILITY_50", "RELIABILITY_60"]:
            if config in scores.columns:
                scores[f"{config}_DELTA"] = (
                    scores[config] - baseline
                )

    print(scores.round(2).to_string())

    print("\n" + "=" * 80)
    print("SUMMARY BY CONFIGURATION")
    print("=" * 80)

    for config in CONFIGS:
        subset = summary.loc[summary["CONFIG"].eq(config)]
        complete = subset.loc[subset["STATUS"].eq("COMPLETE")]

        print(
            f"{config:17s} "
            f"Complete={len(complete):2d} "
            f"AvgActual={complete['ACTUAL'].mean():7.2f} "
            f"Exceptions={subset['EXCEPTIONS_SELECTED'].sum():3.0f}"
        )

    print("\nSaved lineup backtest results.")


if __name__ == "__main__":
    main()
