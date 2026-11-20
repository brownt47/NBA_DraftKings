"""Leakage-safe opponent-by-position statistical impact features.

Run from the NBA_DraftKings repository root:
    python -m src.modeling.opponent_stat_impact

Writes a separate feature file. Does not change production datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = [Path('data/nba_features_2025_26.csv'), Path('data/nba_features_2026_27.csv')]
DIRECTORY = Path('data/espn_player_directory.csv')
OUTPUT = Path('reports/modeling/opponent_stat_impact/features.csv')
STATS = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV']
GROUPS = {'G': 'Guard', 'PG': 'Guard', 'SG': 'Guard',
          'F': 'Forward', 'SF': 'Forward', 'PF': 'Forward', 'C': 'Center'}
PRIOR_STRENGTH = 30.0  # player-game equivalents, toward neutral impact


def build_features():
    frames = [pd.read_csv(p, dtype={'GAME_ID': str, 'PLAYER_ID': str}) for p in DATA]
    df = pd.concat(frames, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.normalize()
    df['SEASON'] = np.where(df['GAME_DATE'].dt.month >= 10,
                            df['GAME_DATE'].dt.year,
                            df['GAME_DATE'].dt.year - 1)
    df = df.sort_values(['GAME_DATE', 'GAME_ID', 'PLAYER_ID']).reset_index(drop=True)
    if df.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-games; inspect input datasets')
    directory = pd.read_csv(DIRECTORY, dtype={'PLAYER_ID': str})[['PLAYER_ID', 'POSITION']]
    if directory['PLAYER_ID'].duplicated().any():
        raise ValueError('Duplicate PLAYER_ID in position directory')
    df = df.merge(directory, on='PLAYER_ID', how='left', validate='many_to_one')
    df['POSITION_GROUP'] = df['POSITION'].map(GROUPS)
    if df['POSITION_GROUP'].isna().any():
        print('Unmapped player-games:', int(df['POSITION_GROUP'].isna().sum()))
    # Pregame expectations: shift by one player-game, then average prior 10.
    # This works even for categories without existing pregame rolling features.
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    for stat in STATS:
        df[stat] = pd.to_numeric(df[stat], errors='coerce')
        df[f'EXPECT_{stat}'] = df.groupby('PLAYER_ID', sort=False)[stat].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )
    # One row per actual opposing team game and position, aggregating
    # player-game residuals. Unavailable expectations contribute no residual.
    eligible = df.loc[df['POSITION_GROUP'].notna()].copy()
    for stat in STATS:
        eligible[f'ERR_{stat}'] = eligible[stat] - eligible[f'EXPECT_{stat}']
        eligible[f'CNT_{stat}'] = eligible[f'ERR_{stat}'].notna().astype(int)
        eligible[f'ERR_{stat}'] = eligible[f'ERR_{stat}'].fillna(0.0)
    agg = {f'ERR_{s}': 'sum' for s in STATS}
    agg.update({f'CNT_{s}': 'sum' for s in STATS})
    # Group by opponent, game, and the offensive player's position.
    games = eligible.groupby(
        ['SEASON', 'OPPONENT', 'POSITION_GROUP', 'GAME_DATE', 'GAME_ID'],
        as_index=False, sort=True
    ).agg(agg)
    # Same-day cutoff: aggregate all games on the date before constructing
    # historical totals. This prevents leakage between same-day games.
    daily = games.groupby(
        ['SEASON', 'OPPONENT', 'POSITION_GROUP', 'GAME_DATE'],
        as_index=False, sort=True
    ).agg(agg)
    daily['TEAM_GAMES'] = games.groupby(
        ['SEASON', 'OPPONENT', 'POSITION_GROUP', 'GAME_DATE']
    ).size().to_numpy()
    daily = daily.sort_values(
        ['SEASON', 'OPPONENT', 'POSITION_GROUP', 'GAME_DATE']
    ).reset_index(drop=True)
    keys = ['SEASON', 'OPPONENT', 'POSITION_GROUP']
    group = daily.groupby(keys, sort=False)
    daily['OPP_POS_PRIOR_GAMES'] = group['TEAM_GAMES'].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    for stat in STATS:
        err, cnt = f'ERR_{stat}', f'CNT_{stat}'
        # Season-to-date uses only earlier dates, resets each season.
        prior_sum = group[err].transform(lambda x: x.cumsum().shift(1).fillna(0))
        prior_n = group[cnt].transform(lambda x: x.cumsum().shift(1).fillna(0))
        daily[f'OPP_POS_{stat}_IMPACT_SEASON'] = prior_sum / (prior_n + PRIOR_STRENGTH)
        # Last 10 *opponent games*, not 10 player records. For the usual
        # one-game-per-team-date schedule this is a rolling 10-game window.
        # Date-level grouping also protects same-day doubleheaders.
        rolling_sum = group[err].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).sum().fillna(0)
        )
        rolling_n = group[cnt].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).sum().fillna(0)
        )
        daily[f'OPP_POS_{stat}_IMPACT_10'] = rolling_sum / (rolling_n + PRIOR_STRENGTH)
        daily[f'OPP_POS_{stat}_N_10'] = rolling_n
    # Build completed-date snapshots, then join strictly to prior dates.
    # built from completed daily totals (including the last completed date).
    snapshots = daily[keys + ['GAME_DATE']].copy()
    grouped = daily.groupby(keys, sort=False)
    snapshots['OPP_POS_PRIOR_GAMES'] = grouped['TEAM_GAMES'].cumsum()
    for stat in STATS:
        err, cnt = f'ERR_{stat}', f'CNT_{stat}'
        snapshots[f'OPP_POS_{stat}_IMPACT_SEASON'] = (
            grouped[err].cumsum() / (grouped[cnt].cumsum() + PRIOR_STRENGTH)
        )
        snapshots[f'OPP_POS_{stat}_IMPACT_10'] = (
            grouped[err].transform(lambda x: x.rolling(10, min_periods=1).sum()) /
            (grouped[cnt].transform(lambda x: x.rolling(10, min_periods=1).sum()) + PRIOR_STRENGTH)
        )
        snapshots[f'OPP_POS_{stat}_N_10'] = grouped[cnt].transform(
            lambda x: x.rolling(10, min_periods=1).sum()
        )
    
        feature_cols = ['OPP_POS_PRIOR_GAMES'] + [
        f'OPP_POS_{stat}_{suffix}'
        for stat in STATS
        for suffix in ('IMPACT_SEASON', 'IMPACT_10', 'N_10')
    ]
    
    left = df[['GAME_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON',
               'OPPONENT', 'POSITION_GROUP']].copy()
    left['_ROW'] = np.arange(len(left))
    result_parts = []
    for key, subset in left.groupby(keys, dropna=False):
        hist = snapshots
        for col, value in zip(keys, key):
            hist = hist.loc[hist[col].eq(value)]
        subset = subset.sort_values('GAME_DATE')
        if hist.empty:
            for col in feature_cols:
                subset[col] = 0.0
            result_parts.append(subset)
        else:
            result_parts.append(pd.merge_asof(
                subset, hist[['GAME_DATE'] + feature_cols].sort_values('GAME_DATE'),
                on='GAME_DATE', direction='backward', allow_exact_matches=False
            ))
    out = pd.concat(result_parts, ignore_index=True).sort_values('_ROW')
    out[feature_cols] = out[feature_cols].fillna(0.0)
    assert len(out) == len(df)
    assert not out.duplicated(['GAME_ID', 'PLAYER_ID']).any()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out[['GAME_ID', 'PLAYER_ID', 'GAME_DATE', 'POSITION_GROUP'] + feature_cols].to_csv(OUTPUT, index=False)
    print(f'Saved {len(out):,} player-games, {len(feature_cols)} opponent features to {OUTPUT}')
    print('Position coverage:', round(100 * df['POSITION_GROUP'].notna().mean(), 2), '%')
    print('Features:', ', '.join(feature_cols))


if __name__ == '__main__':
    build_features()
