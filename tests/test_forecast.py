"""The live ledger: forecasts only before kick-off, never rewritten, fixtures never leak."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_features import synthetic_season  # noqa: E402

from feature_engineering import MODEL_FEATURES, build_features  # noqa: E402
from forecast import (  # noqa: E402
    LEDGER_COLUMNS,
    LEDGER_PATH,
    append_to_ledger,
    check_entries,
    forecast,
    kickoff_utc,
    live_scores,
    load_ledger,
    new_entries,
    season_stem,
    settle,
    upcoming_fixtures,
)
from train_models import make_model, predict_sorted  # noqa: E402

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def forecasts(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Forecast rows of (home, away, kick-off UTC) with flat probabilities."""
    return pd.DataFrame(
        {
            "Season": "E0_2627",
            "HomeTeam": [r[0] for r in rows],
            "AwayTeam": [r[1] for r in rows],
            "KickoffUTC": pd.to_datetime([r[2] for r in rows], utc=True),
            **{f"{p}_{o}": 1 / 3 for p in "PB" for o in "HDA"},
        }
    )


def test_season_rolls_over_in_july():
    assert season_stem(date(2026, 10, 1)) == "E0_2627"
    assert season_stem(date(2027, 5, 20)) == "E0_2627"
    assert season_stem(date(2027, 7, 1)) == "E0_2728"


def test_kickoff_converts_uk_local_time_to_utc():
    fixtures = pd.DataFrame(
        {"Date": ["15/08/2026", "12/12/2026", "26/12/2026"], "Time": ["15:00", "20:00", None]}
    )
    expected = pd.to_datetime(
        ["2026-08-15 14:00", "2026-12-12 20:00", "2026-12-26 00:00"], utc=True
    )
    pd.testing.assert_series_equal(kickoff_utc(fixtures), pd.Series(expected), check_names=False)


def test_only_matches_not_yet_started_are_logged():
    batch = forecasts(
        ("Alpha", "Bravo", "2026-10-03 11:30"),  # already kicked off
        ("Charlie", "Delta", "2026-10-03 12:00"),  # kicking off right now
        ("Bravo", "Alpha", "2026-10-03 14:00"),
    )
    entries = new_entries(batch, load_ledger(Path("missing.csv")), NOW, "v1")
    assert entries[["HomeTeam", "AwayTeam"]].values.tolist() == [["Bravo", "Alpha"]]
    assert entries.columns.tolist() == LEDGER_COLUMNS
    assert entries.loc[0, "LoggedAtUTC"] == "2026-10-03T12:00Z"


def test_logged_rows_are_never_rewritten(tmp_path):
    path = tmp_path / "live.csv"
    first = forecasts(("Alpha", "Bravo", "2026-10-04 14:00"))
    append_to_ledger(new_entries(first, load_ledger(path), NOW, "v1"), path)
    before = path.read_bytes()

    # Same fixture again with different odds and a new model: ignored. A new one: appended.
    later = forecasts(
        ("Alpha", "Bravo", "2026-10-04 14:00"), ("Charlie", "Delta", "2026-10-05 19:00")
    )
    later["P_H"] = 0.9
    append_to_ledger(new_entries(later, load_ledger(path), NOW, "v2"), path)

    after = path.read_bytes()
    assert after.startswith(before)
    ledger = load_ledger(path)
    assert ledger["HomeTeam"].tolist() == ["Alpha", "Charlie"]
    assert ledger.loc[0, "ModelVersion"] == "v1"


def test_unplayed_fixture_adds_no_form_to_the_next_one():
    rng = np.random.default_rng(1)
    history = synthetic_season("E0_2526", "2025-08-09", rng)
    upcoming = pd.DataFrame(
        {
            "Date": ["04/10/2026", "11/10/2026"],
            "HomeTeam": ["Alpha", "Bravo"],
            "AwayTeam": ["Charlie", "Alpha"],
            "SeasonFile": "E0_2526",
        }
    )
    features = build_features(pd.concat([history, upcoming], ignore_index=True))
    second = features.iloc[-1]

    alpha = features.iloc[:-2]
    alpha = alpha[(alpha.HomeTeam == "Alpha") | (alpha.AwayTeam == "Alpha")].tail(4)
    is_home = alpha.HomeTeam == "Alpha"
    scored = np.where(is_home, alpha.HomeGoals, alpha.AwayGoals)
    conceded = np.where(is_home, alpha.AwayGoals, alpha.HomeGoals)
    points = np.select([scored > conceded, scored == conceded], [3, 1], 0)

    # The window holds the unplayed fixture plus the last four results: only those count.
    assert second["AwayPointsL5"] == points.mean()
    # And the unplayed fixture moves no Elo rating.
    first = features.iloc[-2]
    assert second["AwayElo"] == first["HomeElo"]


def test_settle_scores_only_played_matches():
    ledger = forecasts(
        ("Alpha", "Bravo", "2026-10-04 14:00"), ("Charlie", "Delta", "2026-10-05 19:00")
    )
    ledger["P_H"], ledger["P_D"], ledger["P_A"] = 0.6, 0.25, 0.15
    results = pd.DataFrame(
        {
            "SeasonFile": ["E0_2627"],
            "HomeTeam": ["Alpha"],
            "AwayTeam": ["Bravo"],
            "FTHG": [2],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )
    settled = settle(ledger, results)
    assert settled["Result"].tolist()[0] == "H" and pd.isna(settled["Result"].tolist()[1])
    scores = live_scores(settled)
    assert scores["model"]["accuracy"] == 1.0
    np.testing.assert_allclose(scores["model"]["log_loss"], -np.log(0.6))
    np.testing.assert_allclose(scores["bookmaker"]["log_loss"], np.log(3))


def test_a_week_without_league_fixtures_yields_none():
    fixtures = pd.DataFrame(
        {
            "Div": ["EC"],
            "Date": ["29/09/2026"],
            "Time": ["19:45"],
            "HomeTeam": ["Barrow"],
            "AwayTeam": ["Scunthorpe"],
        }
    )
    upcoming = upcoming_fixtures(fixtures, pd.DataFrame(), "E0_2627")
    assert upcoming.empty and "KickoffUTC" in upcoming.columns
    assert new_entries(forecasts(), load_ledger(Path("missing.csv")), NOW, "v1").empty


def valid_entries() -> pd.DataFrame:
    batch = forecasts(
        ("Alpha", "Bravo", "2026-10-04 14:00"), ("Charlie", "Delta", "2026-10-05 19:00")
    )
    batch["P_H"], batch["P_D"], batch["P_A"] = 0.5, 0.3, 0.2
    return new_entries(batch, load_ledger(Path("missing.csv")), NOW, "v1")


def test_valid_forecasts_pass_the_check():
    assert check_entries(valid_entries()) == []


def test_missing_bookmaker_odds_are_allowed():
    entries = valid_entries()
    entries.loc[0, ["B_H", "B_D", "B_A"]] = np.nan
    assert check_entries(entries) == []


@pytest.mark.parametrize(
    ("corrupt", "problem"),
    [
        (lambda e: e.assign(P_H=1.2, P_D=-0.1, P_A=-0.1), "model probability outside [0, 1]"),
        (lambda e: e.assign(P_H=0.5, P_D=0.5, P_A=0.5), "model probabilities do not sum to 1"),
        (lambda e: e.assign(B_H=0.9), "bookmaker probabilities do not sum to 1"),
        (lambda e: e.assign(P_D=np.nan), "model probabilities missing"),
        (lambda e: e.assign(LoggedAtUTC=e["KickoffUTC"]), "forecast logged at or after kick-off"),
        (lambda e: e.assign(AwayTeam=e["HomeTeam"]), "a team playing itself"),
        (lambda e: pd.concat([e, e.head(1)]), "match logged twice"),
    ],
)
def test_invalid_forecasts_are_refused(corrupt, problem):
    assert problem in check_entries(corrupt(valid_entries()))


@pytest.mark.skipif(not LEDGER_PATH.exists(), reason="no live forecasts logged yet")
def test_committed_ledger_is_valid():
    assert check_entries(load_ledger()) == []


def test_live_path_reproduces_training_path():
    """Forecasting a fixture with its result hidden gives the probabilities training sees.

    Guards against training/serving skew: the live code builds features from a partial
    season plus fixtures with no goals, the training code from complete seasons.
    """
    rng = np.random.default_rng(2)
    history = synthetic_season("E0_2324", "2023-08-12", rng)
    season = synthetic_season("E0_2425", "2024-08-10", rng)
    full = build_features(pd.concat([history, season], ignore_index=True))
    model = make_model("Logistic Regression", {"C": 1.0}).fit(full[MODEL_FEATURES], full["Result"])
    hidden = ["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]

    for k in range(2, len(season)):
        fixture = season.iloc[[k]].drop(columns=hidden)
        fixture["KickoffUTC"] = pd.to_datetime(fixture["Date"], dayfirst=True).dt.tz_localize(UTC)
        live = forecast(history, season.iloc[:k], fixture, model)

        trained = full[
            (full["SeasonFile"] == "E0_2425")
            & (full["HomeTeam"] == fixture["HomeTeam"].iloc[0])
            & (full["AwayTeam"] == fixture["AwayTeam"].iloc[0])
        ]
        expected = predict_sorted(model, trained[MODEL_FEATURES])[0]  # A, D, H
        np.testing.assert_allclose(live[["P_A", "P_D", "P_H"]].to_numpy()[0], expected)
