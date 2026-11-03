"""Tests for DraftKings prediction baselines."""

import numpy as np
import pandas as pd

from src.modeling.baselines import evaluate_rolling_baseline


def test_rolling_baseline():
    dates = pd.date_range(
        "2025-10-01", "2026-04-30", freq="D"
    )

    df = pd.DataFrame({
        "GAME_DATE": dates,
        "DK_POINTS": 30.0,
        "DK_POINTS_AVG_LAST_5": 28.0,
    })

    results = evaluate_rolling_baseline(df)

    assert len(results) == 3
    assert results["MONTH"].tolist() == [
        "2026-01", "2026-02", "2026-03"
    ]

    assert np.allclose(results["MAE"], 2.0)
    assert np.allclose(results["RMSE"], 2.0)
    assert np.allclose(results["BIAS"], -2.0)

    assert results["N"].tolist() == [31, 28, 31]