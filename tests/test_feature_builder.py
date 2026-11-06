import pandas as pd
from pandas.testing import assert_frame_equal

from src.dk_features import (
    build_features,
    build_historical_features,
    load_player_games,
    load_team_games,
    PLAYER_FILE,
    TEAM_FILE,
)


def test_shared_builder_matches_historical_wrapper():
    players = load_player_games(PLAYER_FILE)
    teams = load_team_games(TEAM_FILE)

    direct = build_features(players, teams)
    wrapped = build_historical_features(season="2026-27")

    assert_frame_equal(
        direct.reset_index(drop=True),
        wrapped.reset_index(drop=True),
        check_dtype=True,
    )