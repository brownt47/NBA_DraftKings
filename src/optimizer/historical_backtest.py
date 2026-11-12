from pathlib import Path
import re
import pandas as pd

from src.dk_features import load_player_games, load_team_games
from src.pregame import load_mapped_slate, build_pregame_features
from src.modeling.train_production import train_production_model
from src.modeling.predict import predict_features
from src.optimizer.classic import optimize_classic


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "dk_backtest"

THRESHOLDS = [0, 18, 20, 22, 24, 26, 28, 30]
MIN_SALARY = 3500

# Set to None to process every archived slate.
# Set to "2026-11-06" to rerun one slate.
BACKTEST_DATE = "2026-11-06"

TRAINING_FILES = [
    DATA / "nba_features_2025_26.csv",
    DATA / "nba_features_2026_27.csv",
]


def make_projection_slate(slate, predictions):
    """Merge predictions without writing to production directories."""

    columns = [
        "PLAYER_ID",
        "PROJECTED_DK_POINTS",
        "DK_POINTS_AVG_SEASON",
        "DK_STD_LAST_5",
        "DK_STD_LAST_10",
        "MIN_AVG_LAST_5",
        "MIN_STD_LAST_10",
        "PLAYER_MINUTES_SHARE_STD_5",
    ]

    missing = [c for c in columns if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    projected = predictions[columns].copy()
    projected["PLAYER_ID"] = projected["PLAYER_ID"].astype(str)

    result = slate.copy()
    result["ESPN_PLAYER_ID"] = result["ESPN_PLAYER_ID"].astype(str)
    result["SALARY"] = pd.to_numeric(
        result["Salary"], errors="coerce"
    )

    result = result.merge(
        projected,
        left_on="ESPN_PLAYER_ID",
        right_on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )

    result["HAS_HISTORY"] = result["DK_POINTS_AVG_SEASON"].notna()

    result["OPTIMIZER_ELIGIBLE"] = (
        ~result["IS_OUT"].eq(True)
        & result["HAS_HISTORY"]
        & result["PROJECTED_DK_POINTS"].notna()
        & ~result["IS_QUESTIONABLE"].eq(True)
    )

    result["OPTIMIZATION_SCORE"] = result["PROJECTED_DK_POINTS"]

    return result


def load_actual_results(players, date):
    """Load ESPN actuals and verified nonappearances."""

    actual = players.loc[
        pd.to_datetime(players["GAME_DATE"]).eq(pd.Timestamp(date)),
        ["PLAYER_ID", "DK_POINTS"],
    ].copy()

    actual["PLAYER_ID"] = actual["PLAYER_ID"].astype(str)
    actual["DK_POINTS"] = pd.to_numeric(
        actual["DK_POINTS"], errors="coerce"
    )

    if actual["PLAYER_ID"].duplicated().any():
        raise ValueError(f"Duplicate actual player IDs on {date}")

    actual = actual.rename(
        columns={"DK_POINTS": "ACTUAL_DK_POINTS"}
    )

    # Verified DNPs and inactive players are recorded separately
    verified_file = (
        OUTPUT / "verified_nonappearances.csv"
    )

    if verified_file.exists():
        verified = pd.read_csv(
            verified_file,
            dtype={"PLAYER_ID": str},
        )

        verified = verified.loc[
            pd.to_datetime(verified["GAME_DATE"]).eq(
                pd.Timestamp(date)
            )
        ].copy()

        if not verified.empty:
            if verified["PLAYER_ID"].duplicated().any():
                raise ValueError(
                    f"Duplicate verified player IDs on {date}"
                )

            # Never overwrite an existing ESPN actual
            existing = set(actual["PLAYER_ID"])

            verified = verified.loc[
                ~verified["PLAYER_ID"].isin(existing)
            ]

            verified = verified.loc[
                verified["RESULT_STATUS"].isin(
                    ["CONFIRMED_DNP", "INACTIVE"]
                )
            ]

            verified = verified[
                ["PLAYER_ID", "ACTUAL_DK_POINTS"]
            ]

            actual = pd.concat(
                [actual, verified],
                ignore_index=True,
            )

    return actual

def process_slate(salary_file, historical_players, historical_teams):
    match = re.search(
        r"DKSalaries_(\d{4})_(\d{2})_(\d{2})_Main\.csv$",
        salary_file.name,
    )

    if not match:
        raise ValueError(f"Unexpected salary filename: {salary_file}")

    date = "-".join(match.groups())
    tag = "_".join(match.groups())

    print(f"\n{'=' * 70}", flush=True)
    print(f"PROCESSING {date}", flush=True)
    print("=" * 70, flush=True)

    model_path = OUTPUT / "models" / f"catboost_{tag}.joblib"

    train_production_model(
        data_path=TRAINING_FILES,
        model_path=model_path,
        cutoff_date=date,
        model_type="catboost",
    )

    slate = load_mapped_slate(salary_file)

    if pd.Timestamp(slate["GAME_DATE"].iloc[0]) != pd.Timestamp(date):
        raise ValueError(f"Salary date mismatch: {salary_file.name}")

    features = build_pregame_features(
        slate,
        historical_players,
        historical_teams,
    )

    predictions = predict_features(
        features,
        model_path=model_path,
    )

    projection_slate = make_projection_slate(slate, predictions)

    projection_path = (
        OUTPUT / "projections" / f"projections_{tag}.pkl"
    )
    projection_slate.to_pickle(projection_path)

    actual = load_actual_results(historical_players, date)

    rows = []
    details = []

    for threshold in THRESHOLDS:
        print(f"  Optimizing minimum minutes: {threshold}", flush=True)

        try:
            result = optimize_classic(
                projection_slate,
                min_salary=MIN_SALARY,
                min_minutes=threshold,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"    Infeasible or failed: {exc}")
            rows.append({
                "DATE": date,
                "MIN_MINUTES": threshold,
                "STATUS": "FAILED",
                "ERROR_MESSAGE": str(exc),
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

        missing = int(lineup["ACTUAL_DK_POINTS"].isna().sum())

        actual_score = (
            float(lineup["ACTUAL_DK_POINTS"].sum())
            if missing == 0
            else float("nan")
        )

        projected_score = float(result["projected_points"])

        rows.append({
            "DATE": date,
            "MIN_MINUTES": threshold,
            "STATUS": "COMPLETE" if missing == 0 else "MISSING_RESULTS",
            "ERROR_MESSAGE": "",
            "SALARY": result["total_salary"],
            "PROJECTED": projected_score,
            "ACTUAL": actual_score,
            "PROJECTION_ERROR": (
                projected_score - actual_score
                if missing == 0 else float("nan")
            ),
            "MISSING_RESULTS": missing,
            "ELIGIBLE_PLAYERS": int(
                (
                    projection_slate["OPTIMIZER_ELIGIBLE"]
                    & (
                        pd.to_numeric(
                            projection_slate["MIN_AVG_LAST_5"],
                            errors="coerce",
                        ).ge(threshold)
                        if threshold > 0 else True
                    )
                    & projection_slate["SALARY"].ge(MIN_SALARY)
                ).sum()
            ),
        })

        lineup["DATE"] = date
        lineup["MIN_MINUTES"] = threshold
        lineup["ACTUAL_MINUS_PROJECTED"] = (
            lineup["ACTUAL_DK_POINTS"]
            - lineup["PROJECTED_DK_POINTS"]
        )

        details.append(lineup)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT / "results" / f"summary_{tag}.csv",
        index=False,
    )

    if details:
        pd.concat(details, ignore_index=True).to_csv(
            OUTPUT / "lineups" / f"lineups_{tag}.csv",
            index=False,
        )

    return summary


def main():
    for folder in ["models", "projections", "lineups", "results"]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    missing_files = [
        str(path) for path in TRAINING_FILES if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing training feature files: {missing_files}"
        )

    salary_files = sorted(
        (DATA / "dk_salaries").glob("DKSalaries_*_Main.csv")
    )

    if BACKTEST_DATE is not None:
        date_tag = BACKTEST_DATE.replace("-", "_")

    salary_files = [
        path for path in salary_files
        if f"DKSalaries_{date_tag}_Main.csv" == path.name
    ]


    if not salary_files:
        raise FileNotFoundError("No archived salary files found.")

    print(f"Found {len(salary_files)} salary slates")

    historical_players = load_player_games()
    historical_teams = load_team_games()

    summaries = []
    failures = []

    for salary_file in salary_files:
        try:
            summary = process_slate(
                salary_file,
                historical_players,
                historical_teams,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"\nFAILED: {salary_file.name}: {exc}", flush=True)
            failures.append({
                "SALARY_FILE": salary_file.name,
                "ERROR": str(exc),
            })

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)

        combined_path = (
            OUTPUT / "results" / "all_slates_summary.csv"
        )

        if BACKTEST_DATE is not None and combined_path.exists():
            previous = pd.read_csv(combined_path)

            previous = previous.loc[
                ~previous["DATE"].isin(combined["DATE"])
            ]

            combined = pd.concat(
                [previous, combined],
                ignore_index=True,
            )

        combined = combined.sort_values(
            ["DATE", "MIN_MINUTES"]
        ).reset_index(drop=True)

        combined.to_csv(combined_path, index=False)

        print("\nCONSOLIDATED RESULTS")
        print(
            combined[
                ["DATE", "MIN_MINUTES", "STATUS",
                 "PROJECTED", "ACTUAL", "MISSING_RESULTS"]
            ].to_string(index=False)
        )

        print(
            "\nSaved:",
            OUTPUT / "results" / "all_slates_summary.csv",
        )

    if failures:
        failure_path = OUTPUT / "results" / "failed_slates.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"\nFailed slates: {len(failures)}")
        print(f"Details: {failure_path}")

    print("\nHISTORICAL BACKTEST FINISHED")


if __name__ == "__main__":
    main()