from pathlib import Path
import pandas as pd

from src.optimizer.classic import optimize_classic
from src.dk_features import load_player_games

ROOT = Path("data/dk_backtest")
PROJECTIONS = ROOT / "projections"
OUTPUT = ROOT / "results" / "minutes_fine_grid"

THRESHOLDS = [20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24]
MIN_SALARY = 3500
MAX_TEAM = 3

OUTPUT.mkdir(parents=True, exist_ok=True)

games = load_player_games()
games["GAME_DATE"] = pd.to_datetime(
    games["GAME_DATE"]
).dt.strftime("%Y-%m-%d")
games["PLAYER_ID"] = games["PLAYER_ID"].astype(str).str.strip()

records = []
lineup_records = []

for path in sorted(PROJECTIONS.glob("projections_*.pkl")):
    date = path.stem.replace("projections_", "").replace("_", "-")
    slate = pd.read_pickle(path)

    print(f"\n=== {date} ===", flush=True)

    actuals = games.loc[
        games["GAME_DATE"].eq(date),
        ["PLAYER_ID", "DK_POINTS"]
    ].copy()

    if actuals["PLAYER_ID"].duplicated().any():
        raise ValueError(f"Duplicate actual player IDs: {date}")

    actual_map = actuals.set_index("PLAYER_ID")["DK_POINTS"]

    for threshold in THRESHOLDS:
        try:
            result = optimize_classic(
                slate,
                min_salary=MIN_SALARY,
                min_minutes=threshold,
                max_players_per_team=MAX_TEAM,
            )

            lineup = result["lineup"].copy()
            lineup["PLAYER_ID"] = (
                lineup["PLAYER_ID"].astype(str).str.strip()
            )
            lineup["ACTUAL_DK_POINTS"] = (
                lineup["PLAYER_ID"].map(actual_map)
            )

            missing = int(lineup["ACTUAL_DK_POINTS"].isna().sum())

            if missing:
                status = "MISSING_RESULTS"
                actual_score = float("nan")
            else:
                status = "COMPLETE"
                actual_score = float(
                    lineup["ACTUAL_DK_POINTS"].sum()
                )

            lineup_ids = tuple(sorted(lineup["DK_ID"].astype(str)))

            records.append({
                "DATE": date,
                "MIN_MINUTES": threshold,
                "STATUS": status,
                "SALARY": result["total_salary"],
                "PROJECTED": result["projected_points"],
                "ACTUAL": actual_score,
                "MISSING_RESULTS": missing,
                "LINEUP_KEY": "|".join(lineup_ids),
            })

            lineup["DATE"] = date
            lineup["MIN_MINUTES"] = threshold
            lineup_records.append(lineup)

            print(
                f"  {threshold:4.1f} min: "
                f"projected={result['projected_points']:7.2f} "
                f"actual={actual_score:7.2f} "
                f"missing={missing}"
            )

        except (ValueError, RuntimeError) as exc:
            records.append({
                "DATE": date,
                "MIN_MINUTES": threshold,
                "STATUS": "FAILED",
                "ERROR": str(exc),
            })
            print(f"  {threshold:4.1f} min: FAILED ({exc})")

summary = pd.DataFrame(records)
summary.to_csv(OUTPUT / "summary.csv", index=False)

if lineup_records:
    pd.concat(lineup_records, ignore_index=True).to_csv(
        OUTPUT / "lineups.csv", index=False
    )

complete = summary.loc[
    summary["STATUS"].eq("COMPLETE")
].copy()

print("\n=== FINE-GRID SUMMARY ===")

if not complete.empty:
    comparison = complete.groupby("MIN_MINUTES").agg(
        SLATES=("DATE", "nunique"),
        AVG_ACTUAL=("ACTUAL", "mean"),
        AVG_PROJECTED=("PROJECTED", "mean"),
        UNIQUE_LINEUPS=("LINEUP_KEY", "nunique"),
    ).round(2)

    print(comparison.to_string())

    comparison.to_csv(OUTPUT / "comparison.csv")

print(f"\nSaved results to {OUTPUT}")
