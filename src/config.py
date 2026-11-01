
from pathlib import Path


# ---------------------------------------------------------
# NBA SEASON CONFIGURATION
# Update this value when starting a new NBA season.
# ---------------------------------------------------------

NBA_SEASON = "2026_27"


# ---------------------------------------------------------
# DATA DIRECTORIES
# Define locations for input and output files.
# ---------------------------------------------------------

DATA_FOLDER = Path("data")
DK_FOLDER = DATA_FOLDER / "dk_salaries"


# ---------------------------------------------------------
# INPUT FILES
# Season-specific historical data and player mappings.
# ---------------------------------------------------------

ESPN_FILE = DATA_FOLDER / (
    f"nba_player_games_{NBA_SEASON}.csv"
)

TEAM_FILE = DATA_FOLDER / (
    f"nba_team_games_{NBA_SEASON}.csv"
)

MAPPING_FILE = DATA_FOLDER / "dk_espn_player_mapping.csv"
