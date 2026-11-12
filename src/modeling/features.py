"""Select pregame features for DraftKings prediction models."""

PREGAME_FEATURES = [
    'PRIOR_GAMES',
    'DK_POINTS_AVG_SEASON',
    'PTS_AVG_SEASON',
    'REB_AVG_SEASON',
    'AST_AVG_SEASON',
    'DK_POINTS_AVG_LAST_3',
    'PTS_AVG_LAST_3',
    'REB_AVG_LAST_3',
    'AST_AVG_LAST_3',
    'DK_STD_LAST_3',
    'DK_POINTS_AVG_LAST_5',
    'PTS_AVG_LAST_5',
    'REB_AVG_LAST_5',
    'AST_AVG_LAST_5',
    'DK_STD_LAST_5',
    'DK_POINTS_AVG_LAST_7',
    'PTS_AVG_LAST_7',
    'REB_AVG_LAST_7',
    'AST_AVG_LAST_7',
    'DK_STD_LAST_7',
    'DK_POINTS_AVG_LAST_10',
    'PTS_AVG_LAST_10',
    'REB_AVG_LAST_10',
    'AST_AVG_LAST_10',
    'DK_STD_LAST_10',
    'FGA_PER_36',
    'FGA_PER_36_TREND',
    'PLAYER_USAGE_PROXY',
    'DK_PER_MIN_BY_MINUTES',
    'MIN_AVG_SEASON',
    'DK_PER_MIN',
    'MIN_AVG_LAST_3',
    'MIN_STD_LAST_3',
    'MIN_AVG_LAST_5',
    'MIN_STD_LAST_5',
    'MIN_AVG_LAST_7',
    'MIN_STD_LAST_7',
    'MIN_AVG_LAST_10',
    'MIN_STD_LAST_10',
    'MIN_TREND_3',
    'LAST_GAME_MIN',
    'TEAM_CHANGE_FLAG',
    'GAMES_WITH_CURRENT_TEAM',
    'MIN_AVG_CURRENT_TEAM',
    'DK_AVG_CURRENT_TEAM',
    'TEAM_ROTATION_CONCENTRATION',
    'TEAM_ROTATION_CONCENTRATION_AVG_5',
    'PLAYER_MINUTES_SHARE_AVG_5',
    'PLAYER_MINUTES_SHARE_STD_5',
    'PLAYER_MINUTES_SHARE_TREND',
    'DK_EMA_3',
    'DK_EMA_10',
    'DK_MOMENTUM',
    'DK_MOMENTUM_CHANGE',
    'DK_CROSSOVER',
    'DK_WHIPSAW_5',
    'OPPONENT',
    'HOME_AWAY',
    'TEAM_GAMES_LAST_7_DAYS',
    'TEAM_GAMES_LAST_14_DAYS',
    'TEAM_TRAVEL_MILES_LAST_7_DAYS',
    'TEAM_TRAVEL_MILES_LAST_14_DAYS',
    'TEAM_TEAM_PRIOR_GAMES',
    'TEAM_PACE_SEASON',
    'TEAM_PACE_LAST_3',
    'TEAM_PACE_LAST_5',
    'TEAM_PACE_LAST_7',
    'TEAM_PACE_LAST_10',
    'TEAM_OFF_RATING_SEASON',
    'TEAM_OFF_RATING_LAST_3',
    'TEAM_OFF_RATING_LAST_5',
    'TEAM_OFF_RATING_LAST_7',
    'TEAM_OFF_RATING_LAST_10',
    'TEAM_DEF_RATING_SEASON',
    'TEAM_DEF_RATING_LAST_3',
    'TEAM_DEF_RATING_LAST_5',
    'TEAM_DEF_RATING_LAST_7',
    'TEAM_DEF_RATING_LAST_10',
    'OPP_GAMES_LAST_7_DAYS',
    'OPP_GAMES_LAST_14_DAYS',
    'OPP_TRAVEL_MILES_LAST_7_DAYS',
    'OPP_TRAVEL_MILES_LAST_14_DAYS',
    'OPP_TEAM_PRIOR_GAMES',
    'OPP_PACE_SEASON',
    'OPP_PACE_LAST_3',
    'OPP_PACE_LAST_5',
    'OPP_PACE_LAST_7',
    'OPP_PACE_LAST_10',
    'OPP_OFF_RATING_SEASON',
    'OPP_OFF_RATING_LAST_3',
    'OPP_OFF_RATING_LAST_5',
    'OPP_OFF_RATING_LAST_7',
    'OPP_OFF_RATING_LAST_10',
    'OPP_DEF_RATING_SEASON',
    'OPP_DEF_RATING_LAST_3',
    'OPP_DEF_RATING_LAST_5',
    'OPP_DEF_RATING_LAST_7',
    'OPP_DEF_RATING_LAST_10',
    'MATCHUP_PACE_SEASON',
    'MATCHUP_PACE_LAST_3',
    'MATCHUP_PACE_LAST_5',
    'MATCHUP_PACE_LAST_7',
    'MATCHUP_PACE_LAST_10',
    'DAYS_REST',
    'IS_BACK_TO_BACK',
    'GAMES_LAST_7_DAYS',
    'GAMES_LAST_14_DAYS',
    'ROAD_TRIP_GAME_NUMBER',
    'IS_FIRST_HOME_AFTER_TRIP',
    'TRAVEL_MILES',
    'TIMEZONE_CHANGE',
    'TIMEZONES_CROSSED',
    'TRAVEL_MILES_LAST_7_DAYS',
    'TRAVEL_MILES_LAST_14_DAYS',
    'TRAVEL_MILES_PER_RECOVERY_DAY',
    'OPP_DAYS_REST',
    'OPP_IS_BACK_TO_BACK',
    'OPP_TRAVEL_MILES',
    'OPP_TIMEZONE_CHANGE',
    'OPP_TIMEZONES_CROSSED',
    'B2B_MIN_LOAD',
    'B2B_LAST_GAME_MIN',
    'B2B_EXCESS_MIN_LOAD',
    'PLAYER_MINUTES_LAST_7_DAYS',
    'PLAYER_MINUTES_LAST_14_DAYS',
]


def select_model_features(df):
    """Return explicitly approved pregame predictor columns."""
    missing = [col for col in PREGAME_FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return df[PREGAME_FEATURES].copy()

def prepare_model_features(df):
    """Select predictors and prepare categorical columns for LightGBM."""
    features = select_model_features(df)

    categorical_columns = ["OPPONENT", "HOME_AWAY"]

    for column in categorical_columns:
        features[column] = features[column].astype("category")

    return features

CATEGORICAL_FEATURES = [
    "OPPONENT",
    "HOME_AWAY",
]

NUMERIC_FEATURES = [
    feature for feature in PREGAME_FEATURES
    if feature not in CATEGORICAL_FEATURES
]

# Experimental ceiling-aware minutes features.
# Keep these separate until walk-forward validation is complete.

MINUTES_CEILING_FEATURES = [
    "MIN_TREND_RECENT_3",
    "MIN_P90_LAST_10",
    "MIN_HEADROOM_LAST_10",
    "MIN_CEILING_RATIO_LAST_10",
]

PREGAME_FEATURES_CEILING = (
    PREGAME_FEATURES + MINUTES_CEILING_FEATURES
)