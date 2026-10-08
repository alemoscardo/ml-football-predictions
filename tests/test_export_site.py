"""The static site's data: valid probabilities, live vs simulated, and numbers that match."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import export_site  # noqa: E402
from forecast import LEDGER_COLUMNS, TIME_FORMAT  # noqa: E402

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


@pytest.fixture
def in_repo_root():
    previous = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(previous)


def test_matchweeks_follow_each_clubs_games():
    home = pd.Series(["A", "C", "A", "B", "C", "A"])
    away = pd.Series(["B", "D", "C", "D", "B", "D"])
    # Round 1: A-B, C-D. Round 2: A-C, B-D. Round 3: C-B, A-D.
    assert export_site.assign_matchweeks(home, away) == [1, 1, 2, 2, 3, 3]


def test_a_rearranged_match_lands_where_it_is_played():
    # A-B is postponed from round 1 and played after both sides' round-2 games. Each has
    # played once by then, so it is shown with round 2, the round it is played in, and
    # the next round's games are numbered as usual.
    home = pd.Series(["C", "A", "B", "A", "C", "D"])
    away = pd.Series(["D", "C", "D", "B", "B", "A"])
    assert export_site.assign_matchweeks(home, away) == [1, 2, 2, 2, 3, 3]


def test_team_codes_fall_back_to_the_first_letters():
    assert export_site.team_code("Nott'm Forest") == "NFO"
    assert export_site.team_code("Wrexham") == "WRE"


def assert_valid(probs: dict[str, float] | None) -> None:
    assert probs is not None
    values = np.array([probs[o] for o in "HDA"])
    assert ((values >= 0) & (values <= 1)).all()
    assert abs(values.sum() - 1) < 1e-3


@pytest.fixture
def site(in_repo_root):
    return export_site.build(NOW)


def test_every_forecast_is_a_valid_distribution(site):
    assert site["matches"], "the live season should have matches"
    for match in site["matches"]:
        assert match["source"] in {"live", "simulated"}
        assert match["matchweek"] >= 1
        assert_valid(match["model"])
        if match["bookmaker"]:
            assert_valid(match["bookmaker"])
        assert match["pick"] == max("HDA", key=lambda o: match["model"][o])


def test_backtest_matches_the_model_report(site):
    report = json.loads((ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))
    backtest = site["record"]["backtest"]
    assert backtest["matches"] == report["rows"]["test"]
    for source in ("model", "bookmaker"):
        expected = report[source]["test"] if source == "model" else report["bookmaker"]["test"]
        np.testing.assert_allclose(backtest[source]["log_loss"], expected["log_loss"], rtol=1e-9)
        np.testing.assert_allclose(backtest[source]["accuracy"], expected["accuracy"])
    assert len(backtest["gap"]) == len(backtest["dates"]) == backtest["matches"]
    assert sum(band["matches"] for band in backtest["calibration"]) == backtest["matches"]


def test_teams_are_ranked_by_rating(site):
    teams = site["teams"]
    assert [t["rank"] for t in teams] == list(range(1, len(teams) + 1))
    assert [t["elo"] for t in teams] == sorted((t["elo"] for t in teams), reverse=True)
    for team in teams:
        assert team["path"][0] == team["start"] and team["path"][-1] == team["elo"]
        assert set(team["form"]) <= {"W", "D", "L"} and len(team["form"]) <= 5


def test_logged_forecasts_are_live_and_the_rest_simulated(in_repo_root, monkeypatch):
    """A ledger row overrides the simulation; a logged fixture not yet played is pending."""
    played = ("Brentford", "Chelsea", "2026-09-18T19:00Z", "2026-09-16T08:23Z")
    upcoming = ("Arsenal", "Man City", "2026-10-17T11:30Z", "2026-10-08T08:23Z")
    ledger = pd.DataFrame(
        [
            ["E0_2627", home, away, kickoff, logged, "v1", 0.5, 0.3, 0.2, 0.4, 0.3, 0.3]
            for home, away, kickoff, logged in (played, upcoming)
        ],
        columns=LEDGER_COLUMNS,
    )
    for column in ("KickoffUTC", "LoggedAtUTC"):
        ledger[column] = pd.to_datetime(ledger[column], format=TIME_FORMAT, utc=True)
    monkeypatch.setattr(export_site, "load_ledger", lambda: ledger)
    monkeypatch.setattr(export_site, "ledger_commits", lambda: ["a" * 40, "b" * 40])

    site = export_site.build(NOW)
    by_pair = {(m["home"], m["away"]): m for m in site["matches"]}

    live = by_pair[("Brentford", "Chelsea")]
    assert live["source"] == "live" and live["commit"] == "a" * 40
    assert live["model"] == {"H": 0.5, "D": 0.3, "A": 0.2}
    assert live["score"] == [3, 0] and live["loggedAt"] == "2026-09-16T08:23Z"

    pending = by_pair[("Arsenal", "Man City")]
    assert pending["result"] is None and pending["score"] is None
    assert pending["commit"] == "b" * 40 and pending["matchweek"] == 6

    others = [m for pair, m in by_pair.items() if pair not in {played[:2], upcoming[:2]}]
    assert others and all(m["source"] == "simulated" and m["commit"] is None for m in others)
    assert site["meta"]["ledger"] == {"logged": 2, "settled": 1, "pending": 1}
    assert site["record"]["live"]["matches"] == 1


def test_written_json_has_no_nan(site, tmp_path):
    export_site.write(site, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "matches.json",
        "meta.json",
        "record.json",
        "teams.json",
    ]
    for path in tmp_path.iterdir():
        assert "NaN" not in path.read_text(encoding="utf-8")
