
"""Historical NBA schedule and travel-related features."""

import pandas as pd
from zoneinfo import ZoneInfo

# NBA team home locations: latitude, longitude, UTC offset
# Standard-time offsets; daylight saving will be handled later.

TEAM_LOCATIONS = {
    "ATL": (33.7573, -84.3963, -5),
    "BOS": (42.3662, -71.0621, -5),
    "BKN": (40.6826, -73.9754, -5),
    "CHA": (35.2251, -80.8392, -5),
    "CHI": (41.8807, -87.6742, -6),
    "CLE": (41.4965, -81.6882, -5),
    "DAL": (32.7905, -96.8103, -6),
    "DEN": (39.7487, -105.0077, -7),
    "DET": (42.3411, -83.0550, -5),
    "GSW": (37.7680, -122.3877, -8),
    "HOU": (29.7508, -95.3621, -6),
    "IND": (39.7639, -86.1555, -5),
    "LAC": (33.9435, -118.3400, -8),
    "LAL": (34.0430, -118.2673, -8),
    "MEM": (35.1382, -90.0506, -6),
    "MIA": (25.7814, -80.1870, -5),
    "MIL": (43.0451, -87.9172, -6),
    "MIN": (44.9795, -93.2760, -6),
    "NOP": (29.9490, -90.0821, -6),
    "NY": (40.7505, -73.9934, -5),
    "OKC": (35.4634, -97.5151, -6),
    "ORL": (28.5392, -81.3839, -5),
    "PHI": (39.9012, -75.1720, -5),
    "PHX": (33.4457, -112.0712, -7),
    "POR": (45.5316, -122.6668, -8),
    "SAC": (38.5802, -121.4996, -8),
    "SA": (29.4270, -98.4375, -6),
    "TOR": (43.6435, -79.3791, -5),
    "UTA": (40.7683, -111.9011, -7),
    "WAS": (38.8981, -77.0209, -5),
}


TEAM_TIMEZONES = {
    "ATL": "America/New_York",
    "BOS": "America/New_York",
    "BKN": "America/New_York",
    "CHA": "America/New_York",
    "CHI": "America/Chicago",
    "CLE": "America/New_York",
    "DAL": "America/Chicago",
    "DEN": "America/Denver",
    "DET": "America/New_York",
    "GS": "America/Los_Angeles",
    "HOU": "America/Chicago",
    "IND": "America/Indiana/Indianapolis",
    "LAC": "America/Los_Angeles",
    "LAL": "America/Los_Angeles",
    "MEM": "America/Chicago",
    "MIA": "America/New_York",
    "MIL": "America/Chicago",
    "MIN": "America/Chicago",
    "NO": "America/Chicago",
    "NY": "America/New_York",
    "OKC": "America/Chicago",
    "ORL": "America/New_York",
    "PHI": "America/New_York",
    "PHX": "America/Phoenix",
    "POR": "America/Los_Angeles",
    "SAC": "America/Los_Angeles",
    "SA": "America/Chicago",
    "TOR": "America/Toronto",
    "UTAH": "America/Denver",
    "WSH": "America/New_York",
}


# ESPN abbreviation aliases
TEAM_LOCATIONS.update({
    "GS": TEAM_LOCATIONS["GSW"],
    "NO": TEAM_LOCATIONS["NOP"],
    "UTAH": TEAM_LOCATIONS["UTA"],
    "WSH": TEAM_LOCATIONS["WAS"],
})


from math import radians, sin, cos, asin, sqrt


def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two locations in miles."""

    earth_radius = 3958.8

    lat1, lon1, lat2, lon2 = map(
        radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * asin(sqrt(a))


def add_historical_travel_features(team_games):
    """Calculate travel distance between consecutive team games."""

    games = team_games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    # Home games occur at the team's arena.
    # Away games occur at the opponent's arena.
    games["GAME_LOCATION_TEAM"] = games.apply(
        lambda row: (
            row["TEAM"]
            if str(row["HOME_AWAY"]).lower() == "home"
            else row["OPPONENT"]
        ),
        axis=1,
    )

    games["PREVIOUS_LOCATION_TEAM"] = (
        games.groupby("TEAM")["GAME_LOCATION_TEAM"].shift(1)
    )

    def calculate_travel(row):
        previous = row["PREVIOUS_LOCATION_TEAM"]
        current = row["GAME_LOCATION_TEAM"]

        if pd.isna(previous):
            return float("nan")

        lat1, lon1, _ = TEAM_LOCATIONS[previous]
        lat2, lon2, _ = TEAM_LOCATIONS[current]

        return haversine_miles(lat1, lon1, lat2, lon2)

    games["TRAVEL_MILES"] = games.apply(
        calculate_travel, axis=1
    )


    # Time-zone changes between consecutive game locations.
    # Positive = eastward; negative = westward.
    
    def calculate_timezone_change(row):
        previous = row["PREVIOUS_LOCATION_TEAM"]
        current = row["GAME_LOCATION_TEAM"]

        if pd.isna(previous):
            return float("nan")

        game_date = row["GAME_DATE"].date()

        def utc_offset(team):
            timezone = ZoneInfo(TEAM_TIMEZONES[team])
            local_noon = pd.Timestamp(game_date).replace(
                hour=12,
                tzinfo=timezone,
            )
            return local_noon.utcoffset().total_seconds() / 3600

        return utc_offset(current) - utc_offset(previous)
    games["TIMEZONE_CHANGE"] = games.apply(
        calculate_timezone_change, axis=1
        )

    games["TIMEZONES_CROSSED"] = games["TIMEZONE_CHANGE"].abs()


    # Travel accumulated in calendar-day windows, including
    # the trip to the current game's location.
    for window in (7, 14):
        column = f"TRAVEL_MILES_LAST_{window}_DAYS"
        games[column] = 0.0

        for team, group in games.groupby("TEAM", sort=False):
            dates = group["GAME_DATE"]
            distances = group["TRAVEL_MILES"].fillna(0)

            for idx in group.index:
                current_date = dates.loc[idx]
                within_window = (
                    (dates <= current_date)
                    & (dates > current_date - pd.Timedelta(days=window))
                )

                games.loc[idx, column] = distances.loc[within_window].sum()

    # DAYS_REST counts full days off between games.
    # Adding 1 avoids division by zero on back-to-backs.
    games["TRAVEL_MILES_PER_RECOVERY_DAY"] = (
        games["TRAVEL_MILES"] / (games["DAYS_REST"] + 1)
    )

    return games




def add_historical_schedule_features(team_games):
    """Calculate team schedule features for each historical game."""

    games = team_games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    # Days off between team games.
    previous_date = games.groupby("TEAM")["GAME_DATE"].shift(1)
    days_between = (games["GAME_DATE"] - previous_date).dt.days

    games["DAYS_REST"] = days_between - 1
    games["IS_BACK_TO_BACK"] = (days_between == 1).astype(int)

    # Number of previous team games within a calendar-day window.
    for window in (7, 14):
        games[f"GAMES_LAST_{window}_DAYS"] = (
            games.groupby("TEAM")["GAME_DATE"].transform(
                lambda dates: dates.apply(
                    lambda current: (
                        (dates < current)
                        & (dates >= current - pd.Timedelta(days=window))
                    ).sum()
                )
            )
        )

    # Track consecutive away games and first home game after a trip.
    def road_trip_features(group):
        road_number = 0
        road_numbers = []
        first_home = []

        for location in group["HOME_AWAY"]:
            if str(location).lower() == "away":
                road_number += 1
                road_numbers.append(road_number)
                first_home.append(0)
            else:
                first_home.append(int(road_number >= 2))
                road_number = 0
                road_numbers.append(0)

        group = group.copy()
        group["ROAD_TRIP_GAME_NUMBER"] = road_numbers
        group["IS_FIRST_HOME_AFTER_TRIP"] = first_home

        return group

    games = pd.concat(
        [
            road_trip_features(group)
            for _, group in games.groupby("TEAM", sort=False)
        ],
        ignore_index=True
    )

    return games

def add_player_workload_features(players):
    """Calculate player workload interactions using pregame features."""

    players = players.copy()

    players["B2B_MIN_LOAD"] = (
        players["IS_BACK_TO_BACK"]
        * players["MIN_AVG_LAST_5"]
    )

    players["B2B_LAST_GAME_MIN"] = (
        players["IS_BACK_TO_BACK"]
        * players["LAST_GAME_MIN"]
    )

    players["B2B_EXCESS_MIN_LOAD"] = (
        players["IS_BACK_TO_BACK"]
        * (
            players["LAST_GAME_MIN"]
            - players["MIN_AVG_LAST_5"]
        )
    )

    return players


def add_player_recent_minutes(players):
    """Calculate player minutes accumulated before each game."""

    players = players.copy()
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    players = players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    for window in (7, 14):
        column = f"PLAYER_MINUTES_LAST_{window}_DAYS"

        def calculate_minutes(group):
            dates = group["GAME_DATE"]
            minutes = pd.to_numeric(group["MIN"], errors="coerce")

            return pd.Series(
                [
                    minutes.loc[
                        (dates < current)
                        & (dates >= current - pd.Timedelta(days=window))
                    ].sum()
                    for current in dates
                ],
                index=group.index,
            )

        players[column] = (
            players.groupby("PLAYER_ID", group_keys=False)
            .apply(calculate_minutes, include_groups=False)
        )

    return players
