
# ---------------------------------------------------------
# NBA DATA UPDATE CONTROLLER TESTS
# ---------------------------------------------------------

import subprocess

import pytest

from src import update_nba_data


# ---------------------------------------------------------
# SCRAPER EXECUTION ORDER TEST
# ---------------------------------------------------------

def test_update_runs_scrapers_in_order(monkeypatch):

    called = []

    def fake_run_scraper(module_name):
        called.append(module_name)

    monkeypatch.setattr(
        update_nba_data, "run_scraper", fake_run_scraper
    )

    update_nba_data.main()

    assert called == [
        "src.nba_scraper",
        "src.nba_team_scraper"
    ]


# ---------------------------------------------------------
# STOP ON SCRAPER FAILURE TEST
# ---------------------------------------------------------

def test_update_stops_on_failure(monkeypatch):

    called = []

    def fake_run_scraper(module_name):
        called.append(module_name)

        if module_name == "src.nba_scraper":
            raise subprocess.CalledProcessError(
                1, module_name
            )

    monkeypatch.setattr(
        update_nba_data, "run_scraper", fake_run_scraper
    )

    with pytest.raises(subprocess.CalledProcessError):
        update_nba_data.main()

    assert called == ["src.nba_scraper"]
