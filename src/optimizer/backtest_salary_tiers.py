from pathlib import Path
import pandas as pd

from src.dk_features import load_player_games
from src.optimizer.classic import optimize_classic
from src.optimizer.historical_backtest import load_actual_results

ROOT = Path("data/dk_backtest")
OUTPUT = Path("reports/modeling/salary_experiments")
OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "BASELINE": "baseline",
    "NO_7K_9K": "exclude",
    "PENALTY_7K_9K": "penalty",
}

players = load_player_games()
players["PLAYER_ID"] = players["PLAYER_ID"].astype(str)

paths = sorted(
    list((ROOT / "projections").glob("projections_*.pkl"))
    + list((ROOT / "extended" / "projections").glob("projections_*.pkl"))
)

# Prevent duplicate dates from being counted twice.
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

    actual = load_actual_results(players, date)

    for config, mode in CONFIGS.items():
        trial = slate.copy()

        salary = pd.to_numeric(
            trial["SALARY"], errors="coerce"
        )

        minutes = pd.to_numeric(
            trial["MIN_AVG_LAST_5"], errors="coerce"
        )

        # Same eligibility baseline for every configuration.
        trial["OPTIMIZER_ELIGIBLE"] = (
            trial["OPTIMIZER_ELIGIBLE"].eq(True)
            & minutes.ge(23)
        )

        tier = salary.ge(7000) & salary.lt(9000)

        if mode == "exclude":
            trial["OPTIMIZER_ELIGIBLE"] &= ~tier

        if mode == "penalty":
            if "OPTIMIZATION_SCORE" not in trial.columns:
                trial["OPTIMIZATION_SCORE"] = (
                    trial["PROJECTED_DK_POINTS"]
                )

            trial["OPTIMIZATION_SCORE"] = pd.to_numeric(
                trial["OPTIMIZATION_SCORE"],
                errors="coerce"
            ) - 3.0 * tier.astype(float)

        try:
            result = optimize_classic(
                trial,
                min_salary=3500,
                min_minutes=0,
                max_players_per_team=3,
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

            n_mid = int(
                lineup["SALARY"].between(
                    7000, 8999
                ).sum()
            )

            summaries.append({
                "DATE": date,
                "CONFIG": config,
                "STATUS": (
                    "COMPLETE" if missing == 0
                    else "MISSING_RESULTS"
                ),
                "PROJECTED": result["projected_points"],
                "ACTUAL": actual_points,
                "SALARY": result["total_salary"],
                "MID_TIER_COUNT": n_mid,
                "MISSING_RESULTS": missing,
            })

            lineup["DATE"] = date
            lineup["CONFIG"] = config
            details.append(lineup)

            print(
                f"{config:16s} "
                f"Proj={result['projected_points']:7.2f} "
                f"Actual={actual_points:7.2f} "
                f"MidTier={n_mid}"
            )

        except (ValueError, RuntimeError) as exc:
            print(f"{config}: FAILED: {exc}")

            summaries.append({
                "DATE": date,
                "CONFIG": config,
                "STATUS": "FAILED",
                "PROJECTED": float("nan"),
                "ACTUAL": float("nan"),
                "SALARY": float("nan"),
                "MID_TIER_COUNT": float("nan"),
                "MISSING_RESULTS": float("nan"),
            })

summary = pd.DataFrame(summaries)

summary.to_csv(
    OUTPUT / "salary_backtest.csv",
    index=False
)

if details:
    pd.concat(details, ignore_index=True).to_csv(
        OUTPUT / "salary_lineup_details.csv",
        index=False
    )

print("\n=== ACTUAL POINTS BY DATE ===")

scores = summary.pivot(
    index="DATE",
    columns="CONFIG",
    values="ACTUAL"
)

if "BASELINE" in scores:
    for config in CONFIGS:
        if config != "BASELINE":
            scores[f"{config}_DELTA"] = (
                scores[config] - scores["BASELINE"]
            )

print(scores.round(2).to_string())

print("\n=== SUMMARY ===")

for config in CONFIGS:
    subset = summary[
        (summary["CONFIG"] == config)
        & (summary["STATUS"] == "COMPLETE")
    ]

    print(
        f"{config:16s} "
        f"N={len(subset):2d} "
        f"AvgActual={subset['ACTUAL'].mean():7.2f} "
        f"AvgProjected={subset['PROJECTED'].mean():7.2f}"
    )

print("\nSaved results to:", OUTPUT)
