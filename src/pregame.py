
"""Prepare DraftKings NBA Classic slates for pregame predictions."""

import argparse
from pathlib import Path

import pandas as pd

from src.slates.salaries import load_dk_salaries


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MAPPING_FILE = DATA_DIR / "dk_espn_player_mapping.csv"

# DraftKings abbreviations that differ from ESPN's.
TEAM_ALIASES = {
    "GSW": "GS",
    "NOP": "NO",
    "NYK": "NY",
    "UTA": "UTAH",
    "WAS": "WSH",
    "SAS": "SA",
}


def normalize_team(team):
    """Convert DraftKings team abbreviations to ESPN abbreviations."""
    team = str(team).strip().upper()
    return TEAM_ALIASES.get(team, team)


def add_synthetic_game_ids(slate):
    """Assign a unique synthetic game ID to each matchup."""
    slate = slate.copy()

    slate["GAME_ID"] = (
        "PREGAME_"
        + slate["GAME_DATE"].astype(str).str.replace("-", "", regex=False)
        + "_"
        + slate["MATCHUP"].str.replace("@", "_", regex=False)
    )

    return slate



def build_pregame_player_rows(slate, historical_players):
    """Create one pregame player row per DraftKings slate player."""
    rows = pd.DataFrame(
        {
            "GAME_ID": slate["GAME_ID"].to_numpy(),
            "GAME_DATE": pd.to_datetime(slate["GAME_DATE"]).to_numpy(),
            "PLAYER_ID": slate["ESPN_PLAYER_ID"].astype(str).to_numpy(),
            "PLAYER_NAME": slate["DK_NAME"].to_numpy(),
            "TEAM": slate["ESPN_TEAM"].to_numpy(),
            "OPPONENT": slate["ESPN_OPPONENT"].to_numpy(),
            "HOME_AWAY": slate["HOME_AWAY"].to_numpy(),
            "MATCHUP": slate["MATCHUP"].to_numpy(),
        }
    )

    # Match the historical dataset's structure.
    rows = rows.reindex(columns=historical_players.columns)

    # These are not completed games.
    # All actual game statistics must remain missing.
    rows["IS_PREGAME"] = True

    if rows.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate synthetic player-game rows.")

    return rows



def build_pregame_team_rows(slate, historical_teams):
    """Create two pregame team rows for each DraftKings matchup."""

    rows = slate[
        [
            "GAME_ID",
            "GAME_DATE",
            "ESPN_TEAM",
            "ESPN_OPPONENT",
            "HOME_AWAY",
        ]
    ].drop_duplicates(subset=["GAME_ID", "ESPN_TEAM"]).copy()

    rows = rows.rename(
        columns={
            "ESPN_TEAM": "TEAM",
            "ESPN_OPPONENT": "OPPONENT",
        }
    )

    rows["GAME_DATE"] = pd.to_datetime(rows["GAME_DATE"])
    rows["HOME_AWAY"] = rows["HOME_AWAY"].str.lower()

    # NBA regulation game length is known before tipoff.
    rows["GAME_MINUTES"] = 48

    # Match the historical team schema.
    # All unknown game results remain NaN.
    rows = rows.reindex(columns=historical_teams.columns)

    rows["IS_PREGAME"] = True

    if rows.duplicated(["GAME_ID", "TEAM"]).any():
        raise ValueError("Duplicate synthetic team-game rows.")

    game_counts = rows.groupby("GAME_ID")["TEAM"].nunique()

    if not game_counts.eq(2).all():
        raise ValueError("Every synthetic game must have exactly two teams.")

    return rows



def load_mapped_slate(salary_file, mapping_file=MAPPING_FILE):
    """Load one DraftKings slate and attach confirmed ESPN player IDs."""
    slate = load_dk_salaries(salary_file)

    mapping = pd.read_csv(
        mapping_file,
        dtype={"DK_ID": str, "ESPN_PLAYER_ID": str},
    )

    if mapping["DK_ID"].duplicated().any():
        raise ValueError("Duplicate DraftKings IDs in player mapping.")

    slate = slate.merge(
        mapping[["DK_ID", "ESPN_PLAYER_ID"]],
        on="DK_ID",
        how="left",
        validate="one_to_one",
    )

    missing = slate["ESPN_PLAYER_ID"].isna()

    if missing.any():
        names = slate.loc[missing, "DK_NAME"].tolist()
        print(f"WARNING: Skipping unmapped DraftKings players: {names}")
        slate = slate.loc[~missing].copy()

    if slate.empty:
        raise ValueError("No mapped DraftKings players remain.")

    if slate["ESPN_PLAYER_ID"].duplicated().any():
        raise ValueError("Multiple DraftKings players map to one ESPN ID.")

    if slate["GAME_DATE"].nunique() != 1:
        raise ValueError("Salary file contains multiple game dates.")

    slate["ESPN_TEAM"] = slate["TEAM"].map(normalize_team)
    slate["ESPN_OPPONENT"] = slate["OPPONENT"].map(normalize_team)

    slate = add_synthetic_game_ids(slate)

    return slate



def build_pregame_features(slate, historical_players, historical_teams):
    """Generate pregame features using the shared historical pipeline."""
    from src.dk_features import build_features

    pregame_players = build_pregame_player_rows(
        slate, historical_players
    )
    pregame_teams = build_pregame_team_rows(
        slate, historical_teams
    )

    slate_date = pd.Timestamp(slate["GAME_DATE"].iloc[0])

    # Only completed games strictly before the slate date.
    historical_players = historical_players.loc[
        pd.to_datetime(historical_players["GAME_DATE"]) < slate_date
    ].copy()

    historical_teams = historical_teams.loc[
        pd.to_datetime(historical_teams["GAME_DATE"]) < slate_date
    ].copy()

    if historical_players["GAME_ID"].astype(str).isin(
        pregame_players["GAME_ID"]
    ).any():
        raise ValueError("Synthetic player GAME_ID collision.")

    if historical_teams["GAME_ID"].astype(str).isin(
        pregame_teams["GAME_ID"]
    ).any():
        raise ValueError("Synthetic team GAME_ID collision.")

    # Keep the historical rows unmodified.
    historical_players["IS_PREGAME"] = False
    historical_teams["IS_PREGAME"] = False

    # Use consistent player ID types before calculating rolling features.
    historical_players["PLAYER_ID"] = (
        historical_players["PLAYER_ID"].astype(str)
    )

    pregame_players["PLAYER_ID"] = (
        pregame_players["PLAYER_ID"].astype(str)
    )

    players = pd.concat(
        [historical_players, pregame_players],
        ignore_index=True,
    )
    teams = pd.concat(
        [historical_teams, pregame_teams],
        ignore_index=True,
    )


    print("\nPlayer ID diagnostic:")

    historical_ids = set(
        historical_players["PLAYER_ID"].dropna().astype(str)
    )
    pregame_ids = set(
        pregame_players["PLAYER_ID"].dropna().astype(str)
    )

    print("Historical ID dtype:", historical_players["PLAYER_ID"].dtype)
    print("Pregame ID dtype:", pregame_players["PLAYER_ID"].dtype)
    print("Historical unique players:", len(historical_ids))
    print("Pregame unique players:", len(pregame_ids))
    print("Matching IDs:", len(historical_ids & pregame_ids))

    print("\nExample historical IDs:")
    print(historical_players["PLAYER_ID"].head(5).tolist())

    print("\nExample pregame IDs:")
    print(pregame_players["PLAYER_ID"].head(5).tolist())


    features = build_features(players, teams)

    # Retain only the upcoming-game rows.
    result = features.loc[features["IS_PREGAME"]].copy()

    if len(result) != len(slate):
        raise ValueError(
            f"Expected {len(slate)} pregame rows, got {len(result)}."
        )

    return result



def build_projection_slate(slate, predictions):
    """Combine DK salaries, player eligibility, and projections."""

    projection_columns = [
        "PLAYER_ID",
        "PROJECTED_DK_POINTS",
        "DK_POINTS_AVG_SEASON",
        "DK_STD_LAST_5",
        "DK_STD_LAST_10",
        "MIN_AVG_LAST_5",
        "MIN_STD_LAST_10",
        "PLAYER_MINUTES_SHARE_STD_5",
    ]

    projection_data = predictions[
        projection_columns
    ].copy()



    projection_data["PLAYER_ID"] = (
        projection_data["PLAYER_ID"].astype(str)
    )

    result = slate.copy()

    result["SALARY"] = pd.to_numeric(
        result["Salary"], errors="coerce"
    )



    result["ESPN_PLAYER_ID"] = (
        result["ESPN_PLAYER_ID"].astype(str)
    )

    result = result.merge(
        projection_data,
        left_on="ESPN_PLAYER_ID",
        right_on="PLAYER_ID",
        how="left",
        validate="one_to_one",
    )

    result["HAS_HISTORY"] = (
        result["DK_POINTS_AVG_SEASON"].notna()
    )

    result["OPTIMIZER_ELIGIBLE"] = (
        ~result["IS_OUT"]
        & result["HAS_HISTORY"]
        & result["PROJECTED_DK_POINTS"].notna()
    )

    result["VALUE_PER_1000"] = (
        result["PROJECTED_DK_POINTS"]
        / (result["SALARY"] / 1000)
    )

    result["PROJECTION_VS_DK_AVG"] = (
        result["PROJECTED_DK_POINTS"]
        - result["DK_AVG_POINTS"]
    )

    output_dir = Path("data/dk_projections")
    output_dir.mkdir(parents=True, exist_ok=True)

    slate_date = pd.to_datetime(
        result["GAME_DATE"].iloc[0]
    ).strftime("%Y_%m_%d")

    output_file = output_dir / f"projections_{slate_date}_Main.pkl"

    result.to_pickle(output_file)

    print(f"\nSaved projections: {output_file}")

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DraftKings pregame projections."
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Slate date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/catboost_production.joblib"),
        help="Path to the trained production model.",
    )

    args = parser.parse_args()

    slate_date = pd.to_datetime(
        args.date,
        format="%Y-%m-%d",
        errors="raise",
    )

    date_tag = slate_date.strftime("%Y_%m_%d")

    salary_file = (
        DATA_DIR
        / "dk_salaries"
        / f"DKSalaries_{date_tag}_Main.csv"
    )

    if not salary_file.is_file():
        parser.error(
            f"DraftKings salary file not found: {salary_file}"
        )

    slate = load_mapped_slate(salary_file)

    from src.dk_features import load_player_games

    historical_players = load_player_games()

    pregame_players = build_pregame_player_rows(
        slate, historical_players
    )

    from src.dk_features import load_team_games

    historical_teams = load_team_games()

    pregame_teams = build_pregame_team_rows(
        slate, historical_teams
    )

    print(f"Pregame team rows: {len(pregame_teams)}")
    print(f"Unique games: {pregame_teams['GAME_ID'].nunique()}")
    print(f"Missing team points: {pregame_teams['PTS'].isna().sum()}")
    print(f"Missing pace: {pregame_teams['PACE_EST'].isna().sum()}")


    print(f"Pregame player rows: {len(pregame_players)}")
    print(f"Missing actual minutes: {pregame_players['MIN'].isna().sum()}")
    print(f"Missing actual DK points: {pregame_players['DK_POINTS'].isna().sum()}")

    print("\nBuilding pregame features...")

    pregame_features = build_pregame_features(
        slate,
        historical_players,
        historical_teams,
    )

    print(f"Feature rows: {len(pregame_features)}")
    print(f"Feature columns: {len(pregame_features.columns)}")

    check_columns = [
        "DK_POINTS_AVG_SEASON",
        "DK_POINTS_AVG_LAST_10",
        "DK_EMA_10",
        "DAYS_REST",
        "TRAVEL_MILES",
    ]

    print("\nFeature availability:")
    for column in check_columns:
        if column in pregame_features:
            count = pregame_features[column].notna().sum()
            print(f"  {column}: {count}/{len(pregame_features)}")
        else:
            print(f"  {column}: COLUMN MISSING")


    missing_ids = pregame_features.loc[
        pregame_features["DK_POINTS_AVG_SEASON"].isna(),
        "PLAYER_ID",
    ].astype(str)

    missing_players = slate.loc[
        slate["ESPN_PLAYER_ID"].astype(str).isin(missing_ids),
        [
            "DK_NAME",
            "TEAM",
            "STATUS",
            "IS_OUT",
            "DK_AVG_POINTS",
            "ESPN_PLAYER_ID",
        ],
    ].copy()

    print("\nPlayers without historical scoring features:")
    print(
        missing_players.sort_values(
            ["IS_OUT", "DK_AVG_POINTS"],
            ascending=[True, False],
        ).to_string(index=False)
    )


    from src.modeling.features import PREGAME_FEATURES

    expected = list(PREGAME_FEATURES)

    missing_columns = [
        column for column in expected
        if column not in pregame_features.columns
    ]

    available_columns = [
        column for column in expected
        if column in pregame_features.columns
    ]

    print("\nModel predictor audit:")
    print(f"Expected predictors: {len(expected)}")
    print(f"Available predictors: {len(available_columns)}")
    print(f"Missing predictor columns: {len(missing_columns)}")

    if missing_columns:
        print("\nMissing columns:")
        for column in missing_columns:
            print(f"  {column}")

    print("\nPredictors with the most missing values:")

    missing_counts = (
        pregame_features[available_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing_counts.head(20).to_string())



    print(f"Slate date: {slate['GAME_DATE'].iloc[0]}")
    print(f"Players: {len(slate)}")
    print(f"Games: {slate['MATCHUP'].nunique()}")
    print(f"Matched ESPN IDs: {slate['ESPN_PLAYER_ID'].notna().sum()}")

    print("\nSynthetic game IDs:")
    games = slate[["MATCHUP", "GAME_ID"]].drop_duplicates()

    for _, game in games.sort_values("MATCHUP").iterrows():
        print(f"  {game['MATCHUP']}: {game['GAME_ID']}")



    from src.modeling.predict import predict_features

    projected = predict_features(
        pregame_features,
        model_path=args.model,
    )

    print(f"\nPrediction diagnostic ({args.model.name}):")
    print(f"Predicted players: {len(projected)}")
    print(
        "Missing predictions:",
        projected["PROJECTED_DK_POINTS"].isna().sum()
    )

    print(
        projected[
            ["PLAYER_NAME", "TEAM", "PROJECTED_DK_POINTS"]
        ]
        .sort_values(
            "PROJECTED_DK_POINTS", ascending=False
        )
        .head(15)
        .to_string(index=False)
    )


    projection_slate = build_projection_slate(
        slate, projected
    )

    print("\nOptimizer eligibility:")
    print(
        projection_slate["OPTIMIZER_ELIGIBLE"]
        .value_counts()
    )

    print("\nTop projected values:")
    print(
        projection_slate.loc[
            projection_slate["OPTIMIZER_ELIGIBLE"],
            [
                "DK_NAME",
                "SALARY",
                "PROJECTED_DK_POINTS",
                "VALUE_PER_1000",
                "STATUS",
            ],
        ]
        .sort_values("VALUE_PER_1000", ascending=False)
        .head(20)
        .to_string(index=False)
    )


    print("\nPosition eligibility examples:")
    print(
        projection_slate[
            ["DK_NAME", "ELIGIBLE_POSITIONS"]
        ].head(15).to_string(index=False)
    )

    
    from src.optimizer.classic import optimize_classic

    optimized = optimize_classic(projection_slate)

    lineup = optimized["lineup"]

    print("\nOPTIMAL DRAFTKINGS CLASSIC LINEUP")
    print(
        lineup[
            [
                "ROSTER_SLOT",
                "DK_NAME",
                "TEAM",
                "SALARY",
                "DK_AVG_POINTS",
                "PROJECTED_DK_POINTS",
                "VALUE_PER_1000",
                "STATUS",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal salary: ${optimized['total_salary']:,}"
    )
    print(
        f"Remaining salary: ${optimized['remaining_salary']:,}"
    )
    print(
        f"Projected DK points: "
        f"{optimized['projected_points']:.2f}"
    )



    # Build a second lineup excluding questionable players.
    safe_slate = projection_slate.copy()

    safe_slate["OPTIMIZER_ELIGIBLE"] = (
        safe_slate["OPTIMIZER_ELIGIBLE"]
        & ~safe_slate["IS_QUESTIONABLE"]
    )

    safe_result = optimize_classic(safe_slate)
    safe_lineup = safe_result["lineup"]

    print("\nLINEUP EXCLUDING QUESTIONABLE PLAYERS")
    print(
        safe_lineup[
            [
                "ROSTER_SLOT",
                "DK_NAME",
                "TEAM",
                "SALARY",
                "DK_AVG_POINTS",
                "PROJECTED_DK_POINTS",
                "VALUE_PER_1000",
                "STATUS",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal salary: ${safe_result['total_salary']:,}"
    )
    print(
        f"Projected DK points: "
        f"{safe_result['projected_points']:.2f}"
    )

    print(
        f"Projection difference: "
        f"{optimized['projected_points'] - safe_result['projected_points']:.2f}"
    )
