"""Guards against look-ahead leakage in the pre-match features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering import MODEL_FEATURES, build_features  # noqa: E402
from train_models import split_seasons  # noqa: E402


def synthetic_season(season: str, start: str, rng: np.random.Generator) -> pd.DataFrame:
    """A double round-robin between four teams with random scores and odds."""
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    dates = pd.date_range(start, periods=len(fixtures), freq="7D")
    home_goals = rng.integers(0, 4, len(fixtures))
    away_goals = rng.integers(0, 4, len(fixtures))
    return pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "HomeTeam": [h for h, _ in fixtures],
            "AwayTeam": [a for _, a in fixtures],
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": np.select([home_goals > away_goals, home_goals == away_goals], ["H", "D"], "A"),
            "HS": rng.integers(5, 20, len(fixtures)),
            "AS": rng.integers(5, 20, len(fixtures)),
            "HST": rng.integers(0, 8, len(fixtures)),
            "AST": rng.integers(0, 8, len(fixtures)),
            "B365H": rng.uniform(1.5, 4.0, len(fixtures)),
            "B365D": rng.uniform(3.0, 4.0, len(fixtures)),
            "B365A": rng.uniform(1.5, 5.0, len(fixtures)),
            "SeasonFile": season,
        }
    )


@pytest.fixture
def raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.concat(
        [
            synthetic_season("E0_2324", "2023-08-12", rng),
            synthetic_season("E0_2425", "2024-08-10", rng),
        ],
        ignore_index=True,
    )


def test_changing_a_result_leaves_its_own_features_untouched(raw):
    target = 15
    baseline = build_features(raw)
    tampered = raw.copy()
    tampered.loc[target, ["FTHG", "FTAG", "FTR", "HST", "AST"]] = [9, 0, "H", 20, 0]
    changed = build_features(tampered)

    row = baseline.index[
        baseline["Date"] == pd.to_datetime(raw.loc[target, "Date"], dayfirst=True)
    ][0]
    before = baseline.loc[:row, MODEL_FEATURES]
    after = changed.loc[:row, MODEL_FEATURES]
    pd.testing.assert_frame_equal(before, after)  # this match and every earlier one
    assert not baseline.loc[row + 1 :, MODEL_FEATURES].equals(
        changed.loc[row + 1 :, MODEL_FEATURES]
    )


def test_rolling_points_use_only_previous_matches(raw):
    features = build_features(raw)
    team = "Alpha"
    games = features[(features.HomeTeam == team) | (features.AwayTeam == team)]
    is_home = games.HomeTeam == team
    scored = np.where(is_home, games.HomeGoals, games.AwayGoals)
    conceded = np.where(is_home, games.AwayGoals, games.HomeGoals)
    points = pd.Series(
        np.select([scored > conceded, scored == conceded], [3, 1], 0), index=games.index
    )
    expected = points.shift(1).rolling(5, min_periods=2).mean()
    actual = np.where(is_home, games.HomePointsL5, games.AwayPointsL5)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_implied_probabilities_sum_to_one(raw):
    features = build_features(raw)
    totals = features[["NormProbHome_B365", "NormProbDraw_B365", "NormProbAway_B365"]].sum(axis=1)
    np.testing.assert_allclose(totals, 1.0)


def test_season_split_is_chronological():
    seasons = pd.DataFrame(
        {"SeasonFile": [f"E0_{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2018, 2024)]}
    )
    split = split_seasons(seasons)
    order = split["burn_in"] + split["train"] + split["validation"] + split["test"]
    assert order == sorted(seasons["SeasonFile"])
    assert split["test"] == ["E0_2324"] and split["validation"] == ["E0_2223"]


def test_post_match_elo_is_display_only(raw):
    """Post-match ratings include the result itself, so they must never be features."""
    assert not any(feature.endswith("After") for feature in MODEL_FEATURES)
    features = build_features(raw)
    # Within a season, a side's next pre-match rating is its last post-match one.
    season = features[features["SeasonFile"] == "E0_2324"]
    for team in ("Alpha", "Delta"):
        games = season[(season.HomeTeam == team) | (season.AwayTeam == team)]
        is_home = games.HomeTeam == team
        before = np.where(is_home, games.HomeElo, games.AwayElo)
        after = np.where(is_home, games.HomeEloAfter, games.AwayEloAfter)
        np.testing.assert_allclose(before[1:], after[:-1])
