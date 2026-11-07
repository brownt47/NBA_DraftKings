from config import (
    DATA_FOLDER,
    DK_FOLDER,
    ESPN_FILE,
    TEAM_FILE,
    MAPPING_FILE,
)
from data_processing.load_data import (
    load_player_games,
    load_team_games,
    load_slate_data,
)
from data_processing.slate import (
    merge_player_mapping,
    build_output_path,
    select_salary_file,
)
from data_processing.player_features import add_player_features
from features.team import calculate_team_features
from features.matchup import (
    add_matchup_features,
    validate_matchup_features,
)
from reporting import (
    print_data_summary,
    print_team_summary,
    print_analysis_summary,
)
from validation import validate_player_features

# ---------------------------------------------------------
# SLATE SELECTION AND CONFIGURATION
# Select the DraftKings salary file for analysis.
# File locations and NBA season are defined in config.py.
# ---------------------------------------------------------

DK_FILE, SLATE_DATE = select_salary_file(DK_FOLDER)

def main():
    print("DraftKings Player Analysis")
    print("=" * 40)
    print(f"Salary file: {DK_FILE.name}")

    # ---------------------------------------------------------
    # DATA LOADING
    # Load salary, mapping, and historical game records.
    # Historical records are filtered before the slate date.
    # ---------------------------------------------------------

    # Load data
    dk, mapping = load_slate_data(
        DK_FILE,
        MAPPING_FILE
    )

    games = load_player_games(ESPN_FILE, SLATE_DATE)

    # Load historical team games
    team_games = load_team_games(
        TEAM_FILE,
        SLATE_DATE
    )

    print_data_summary(SLATE_DATE, games, team_games)

    # ---------------------------------------------------------
    # TEAM FEATURE ENGINEERING
    # Calculate historical team pace and efficiency ratings.
    # ---------------------------------------------------------
    
    team_features = calculate_team_features(team_games)

    print_team_summary(team_features)

    # ---------------------------------------------------------
    # SLATE PREPARATION AND MATCHUPS
    # Connect DraftKings players to ESPN IDs and opponents.
    # Add team matchup statistics and validate matchups.
    # ---------------------------------------------------------
    
    players = merge_player_mapping(dk, mapping)
    players = add_matchup_features(players, team_features)
    validate_matchup_features(players)
 
    # ---------------------------------------------------------
    # PLAYER FEATURE ENGINEERING
    # Calculate season, rolling, minutes, and momentum stats.
    # Add fantasy points per $1,000 of salary.
    # ---------------------------------------------------------
      
    players = add_player_features(players, games)
    validate_player_features(players)

    # ---------------------------------------------------------
    # OUTPUT AND REPORTING
    # Sort players, save the analysis, and print a summary.
    # ---------------------------------------------------------

    # Sort by salary
    players = players.sort_values(
        "Salary", ascending=False
    )

    # Save analysis dataset
    output = build_output_path(
        DATA_FOLDER,
        SLATE_DATE,
        DK_FILE
    )
    players.to_csv(output, index=False)

    print_analysis_summary(players, output)


if __name__ == "__main__":
    main()
