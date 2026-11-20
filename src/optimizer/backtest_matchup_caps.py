from pathlib import Path
import pandas as pd

from src.dk_features import load_player_games
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path("data/dk_backtest")
OUTPUT = Path("reports/modeling/matchup_experiments")
OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "BASELINE": None,
    "GAME_CAP_5": 5,
    "GAME_CAP_4": 4,
    "GAME_CAP_3": 3,
}

players = load_player_games()
players["PLAYER_ID"] = players["PLAYER_ID"].astype(str)

paths = (
    list((ROOT / "projections").glob("projections_*.pkl"))
    + list((ROOT / "extended" / "projections").glob("projections_*.pkl"))
)

paths_by_date = {}

for path in paths:
    date = path.stem.replace("projections_", "").replace("_", "-")
    if date in paths_by_date:
        raise ValueError(f"Duplicate projection date: {date}")
    paths_by_date[date] = path

summaries = []
details = []

for date, path in sorted(paths_by_date.items()):
    print(f"\n=== {date} ===")

    slate = pd.read_pickle(path).copy()
    slate["PLAYER_ID"] = slate["PLAYER_ID"].astype(str)

    minutes = pd.to_numeric(
        slate["MIN_AVG_LAST_5"], errors="coerce"
    )

    slate["OPTIMIZER_ELIGIBLE"] = (
        slate["OPTIMIZER_ELIGIBLE"].eq(True)
        & minutes.ge(23)
    )

    actual = load_actual_results(players, date)

    for config, cap in CONFIGS.items():
        try:
            result = optimize_classic(
                slate.copy(),
                min_salary=3500,
                min_minutes=0,
                max_players_per_team=3,
                max_players_per_matchup=cap,
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

            max_stack = int(
                lineup["MATCHUP"].value_counts().max()
            )

            summaries.append({
                "DATE": date,
                "CONFIG": config,
                "STATUS": "COMPLETE" if missing == 0 else "MISSING_RESULTS",
                "PROJECTED": result["projected_points"],
                "ACTUAL": actual_points,
                "SALARY": result["total_salary"],
                "MAX_GAME_STACK": max_stack,
                "GAMES_USED": lineup["MATCHUP"].nunique(),
                "MISSING_RESULTS": missing,
            })

            lineup["DATE"] = date
            lineup["CONFIG"] = config
            details.append(lineup)

            print(
                f"{config:12s} "
                f"Proj={result['projected_points']:7.2f} "
                f"Actual={actual_points:7.2f} "
                f"Stack={max_stack} "
                f"Games={lineup['MATCHUP'].nunique()}"
            )

        except (ValueError, RuntimeError) as exc:
            print(f"{config}: FAILED ({exc})")
            summaries.append({
                "DATE": date,
                "CONFIG": config,
                "STATUS": "FAILED",
                "PROJECTED": float("nan"),
                "ACTUAL": float("nan"),
                "SALARY": float("nan"),
                "MAX_GAME_STACK": float("nan"),
                "GAMES_USED": float("nan"),
                "MISSING_RESULTS": float("nan"),
            })

summary = pd.DataFrame(summaries)
summary.to_csv(OUTPUT / "matchup_backtest.csv", index=False)

if details:
    pd.concat(details, ignore_index=True).to_csv(
        OUTPUT / "matchup_lineup_details.csv",
        index=False
    )

scores = summary.pivot(
    index="DATE",
    columns="CONFIG",
    values="ACTUAL"
)

for config in CONFIGS:
    if config != "BASELINE":
        scores[f"{config}_DELTA"] = (
            scores[config] - scores["BASELINE"]
        )

print("\n=== ACTUAL POINTS BY DATE ===")
print(scores.round(2).to_string())

print("\n=== SUMMARY ===")

baseline = summary[
    summary["CONFIG"] == "BASELINE"
].set_index("DATE")

for config in CONFIGS:
    subset = summary[
        (summary["CONFIG"] == config)
        & (summary["STATUS"] == "COMPLETE")
    ].copy()

    paired = subset.join(
        baseline[["ACTUAL", "PROJECTED"]],
        on="DATE",
        rsuffix="_BASE",
    ).dropna(subset=["ACTUAL", "ACTUAL_BASE"])

    delta = paired["ACTUAL"] - paired["ACTUAL_BASE"]

    print(
        f"{config:12s} "
        f"N={len(subset):2d} "
        f"AvgActual={subset['ACTUAL'].mean():7.2f} "
        f"AvgProj={subset['PROJECTED'].mean():7.2f} "
        f"AvgDelta={delta.mean():+7.2f} "
        f"Improved={(delta > 0).sum():2d} "
        f"Worse={(delta < 0).sum():2d} "
        f"Same={(delta == 0).sum():2d}"
    )

print("\nSaved:", OUTPUT)
