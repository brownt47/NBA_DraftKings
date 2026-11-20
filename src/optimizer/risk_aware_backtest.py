from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.dk_features import load_player_games
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "dk_backtest" / "risk_aware"
OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "BASE_22": (22, False),
    "BASE_23": (23, False),
    "RISK_22": (22, True),
    "RISK_23": (23, True),
}

MIN_SALARY = 3500
MAX_PLAYERS_PER_TEAM = 3

def get_archives():
    paths = sorted(
        (DATA / "dk_backtest").rglob("projections_*.pkl")
    )
    if not paths:
        raise RuntimeError("No archived projections found")
    return paths


def get_pregame_ratios(players, date):
    history = players.loc[
        players["GAME_DATE"] < pd.Timestamp(date),
        ["PLAYER_ID", "GAME_DATE", "MIN"],
    ].copy()

    history["MIN"] = pd.to_numeric(
        history["MIN"], errors="coerce"
    )
    history = history.dropna(subset=["MIN"])
    history = history.sort_values(
        ["PLAYER_ID", "GAME_DATE"]
    )

    last10 = history.groupby("PLAYER_ID").tail(10)

    summary = last10.groupby("PLAYER_ID").agg(
        MIN_AVG_10_REBUILT=("MIN", "mean"),
        HISTORY_GAMES_10=("MIN", "size"),
    )

    last5 = history.groupby("PLAYER_ID").tail(5)
    summary["MIN_AVG_5_REBUILT"] = (
        last5.groupby("PLAYER_ID")["MIN"].mean()
    )

    summary["MIN_RATIO_5_10"] = (
        summary["MIN_AVG_5_REBUILT"]
        / summary["MIN_AVG_10_REBUILT"].replace(0, np.nan)
    )

    summary.loc[
        summary["HISTORY_GAMES_10"] < 10,
        "MIN_RATIO_5_10"
    ] = np.nan

    return summary[["MIN_RATIO_5_10"]].reset_index()


def prepare_slate(original, ratios):
    slate = original.copy()
    slate["PLAYER_ID"] = slate["PLAYER_ID"].astype(str)

    if slate["PLAYER_ID"].duplicated().any():
        raise ValueError("Duplicate player IDs in slate")

    if "MIN_RATIO_5_10" in slate.columns:
        slate = slate.drop(columns=["MIN_RATIO_5_10"])

    slate = slate.merge(
        ratios,
        on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )

    if len(slate) != len(original):
        raise ValueError("Historical join changed slate size")

    for col in [
        "MIN_AVG_LAST_5",
        "MIN_STD_LAST_10",
        "MIN_RATIO_5_10",
        "PROJECTED_DK_POINTS",
    ]:
        slate[col] = pd.to_numeric(
            slate[col], errors="coerce"
        )

    if "OPTIMIZATION_SCORE" not in slate.columns:
        slate["OPTIMIZATION_SCORE"] = (
            slate["PROJECTED_DK_POINTS"]
        )

    slate["OPTIMIZATION_SCORE"] = pd.to_numeric(
        slate["OPTIMIZATION_SCORE"], errors="coerce"
    )

    minutes = slate["MIN_AVG_LAST_5"]
    rising = slate["MIN_RATIO_5_10"] > 1.05
    volatile = slate["MIN_STD_LAST_10"] > 5.7

    penalty = np.select(
        [
            rising & minutes.ge(20) & minutes.lt(25),
            rising & minutes.ge(25) & minutes.lt(30),
        ],
        [2.0, 1.0],
        default=0.0,
    )

    penalty += np.where(
        rising & volatile & minutes.ge(20) & minutes.lt(30),
        1.0,
        0.0,
    )

    slate["RISK_PENALTY"] = penalty
    return slate


def main():
    players = load_player_games().copy()
    players["GAME_DATE"] = pd.to_datetime(
        players["GAME_DATE"]
    )
    players["PLAYER_ID"] = players["PLAYER_ID"].astype(str)

    if players.duplicated(["PLAYER_ID", "GAME_DATE"]).any():
        raise ValueError("Duplicate historical player-games")

    summaries = []
    details = []

    archives = get_archives()
    print(f"Found {len(archives)} archived slates")

    for path in archives:
        match = re.search(
            r"projections_(\d{4})_(\d{2})_(\d{2})\.pkl$",
            path.name,
        )
        if not match:
            continue

        date = "-".join(match.groups())
        print(f"\n{'=' * 65}")
        print(f"SLATE {date}")
        print("=" * 65)

        original = pd.read_pickle(path)
        ratios = get_pregame_ratios(players, date)
        slate = prepare_slate(original, ratios)

        eligible = slate["OPTIMIZER_ELIGIBLE"].fillna(False)
        candidates = slate.loc[
            eligible & slate["MIN_AVG_LAST_5"].ge(22)
        ]

        print(f"Players: {len(slate)}")
        print(
            f"Ratio coverage among 22+ candidates: "
            f"{candidates['MIN_RATIO_5_10'].notna().mean():.1%}"
        )
        print(
            f"Penalized candidates: "
            f"{int(candidates['RISK_PENALTY'].gt(0).sum())}"
        )

        actual = load_actual_results(players, date)

        for config, (floor, use_risk) in CONFIGS.items():
            trial = slate.copy()

            if use_risk:
                trial["OPTIMIZATION_SCORE"] = (
                    trial["OPTIMIZATION_SCORE"]
                    - trial["RISK_PENALTY"]
                )

            try:
                result = optimize_classic(
                    trial,
                    min_salary=MIN_SALARY,
                    min_minutes=floor,
                    max_players_per_team=MAX_PLAYERS_PER_TEAM,
                )
            except (ValueError, RuntimeError) as exc:
                print(f"  {config}: FAILED ({exc})")
                summaries.append({
                    "DATE": date,
                    "CONFIG": config,
                    "STATUS": "FAILED",
                    "ERROR": str(exc),
                })
                continue

            lineup = result["lineup"].copy()
            lineup["PLAYER_ID"] = (
                lineup["PLAYER_ID"].astype(str)
            )

            lineup = lineup.merge(
                actual,
                on="PLAYER_ID",
                how="left",
                validate="one_to_one",
            )

            missing = int(
                lineup["ACTUAL_DK_POINTS"].isna().sum()
            )
            actual_total = (
                float(lineup["ACTUAL_DK_POINTS"].sum())
                if missing == 0 else np.nan
            )

            penalties = lineup["RISK_PENALTY"].sum()

            summaries.append({
                "DATE": date,
                "CONFIG": config,
                "STATUS": "COMPLETE" if missing == 0 else "INCOMPLETE",
                "ACTUAL_DK_POINTS": actual_total,
                "PROJECTED_DK_POINTS": result["projected_points"],
                "OPTIMIZATION_SCORE": result["optimization_score"],
                "SALARY": result["total_salary"],
                "MISSING_ACTUALS": missing,
                "PENALIZED_PLAYERS": int(
                    lineup["RISK_PENALTY"].gt(0).sum()
                ),
                "TOTAL_RISK_PENALTY": float(penalties),
            })

            lineup["DATE"] = date
            lineup["CONFIG"] = config
            details.append(lineup)

            score = (
                f"{actual_total:.2f}"
                if missing == 0 else "INCOMPLETE"
            )

            print(
                f"  {config:8s} actual={score:>10s} "
                f"projected={result['projected_points']:.2f} "
                f"risk={penalties:.1f} "
                f"missing={missing}"
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    if details:
        pd.concat(details, ignore_index=True).to_csv(
            OUTPUT / "lineups.csv", index=False
        )

    complete = summary.loc[
        summary["STATUS"].eq("COMPLETE")
    ]

    pivot = complete.pivot(
        index="DATE",
        columns="CONFIG",
        values="ACTUAL_DK_POINTS",
    )

    print("\nACTUAL DK POINTS BY SLATE")
    print(pivot.round(2).to_string())

    if all(c in pivot.columns for c in CONFIGS):
        paired = pivot.dropna(subset=list(CONFIGS))
        print(f"\nPAIRED COMPLETE SLATES: {len(paired)}")

        print("\nAVERAGE ACTUAL DK POINTS")
        print(paired.mean().sort_values(ascending=False).round(2))

        baseline = paired["BASE_22"]

        print("\nVERSUS BASE_22")
        for config in ["BASE_23", "RISK_22", "RISK_23"]:
            delta = paired[config] - baseline
            print(
                f"{config:8s} "
                f"avg={delta.mean():+.2f} "
                f"better={(delta > 0).sum()} "
                f"worse={(delta < 0).sum()} "
                f"tied={(delta == 0).sum()}"
            )

    print(f"\nSaved reports to {OUTPUT}")


if __name__ == "__main__":
    main()
