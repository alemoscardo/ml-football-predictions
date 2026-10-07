"""Pre-match features for Premier League outcome models.

Every feature is computed only from information available before kick-off:
team strength (Elo), recent form from earlier matches and rest days. Nothing
from the match being predicted is used. Bookmaker odds are turned into implied
probabilities for the benchmark only; they are not model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RENAME_MAP = {
    "FTHG": "HomeGoals",
    "FTAG": "AwayGoals",
    "HS": "HomeShots",
    "AS": "AwayShots",
    "HST": "HomeShotsTarget",
    "AST": "AwayShotsTarget",
    "FTR": "Result",
}

FORM_WINDOW = 5

# Elo: standard logistic scale, home advantage in rating points, and a pull
# towards the mean between seasons so ratings do not drift.
ELO_START = 1500.0
ELO_NEW_TEAM = 1420.0  # promoted sides are, on average, weaker than the league
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_REGRESSION = 0.2

# Rolling per-team statistics, each averaged over the last FORM_WINDOW matches.
ROLLING_STATS = ["Points", "GoalsFor", "GoalsAgainst", "ShotsTargetFor", "ShotsTargetAgainst"]

TEAM_FEATURES = [
    f"{side}{stat}"
    for side in ("Home", "Away")
    for stat in ["Elo", *[f"{s}L{FORM_WINDOW}" for s in ROLLING_STATS], "RestDays"]
]

DIFF_FEATURES = [
    "EloDiff",
    "EloHomeWinProb",
    "FormPointsDiff",
    "GoalDiffDiff",
    "ShotsTargetDiffDiff",
]

MODEL_FEATURES = TEAM_FEATURES + DIFF_FEATURES

# Bookmaker benchmark only, never fed to the model. Outcome code → column.
BOOKMAKER_PROBS = {"H": "NormProbHome_B365", "D": "NormProbDraw_B365", "A": "NormProbAway_B365"}


def prepare_matches_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse dates, sort chronologically and add implied odds."""
    df = raw_df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in raw_df.columns})
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)

    implied = 1 / df[["B365H", "B365D", "B365A"]].replace(0, np.nan)
    implied = implied.div(implied.sum(axis=1), axis=0)  # strip the bookmaker margin
    df[list(BOOKMAKER_PROBS.values())] = implied.to_numpy()
    return df


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo for both sides, updated after each result.

    Post-match ratings are kept too, for display only: they include the match's own
    result, so they must never be model features.
    """
    ratings: dict[str, float] = {}
    first_season = df["SeasonFile"].iloc[0]
    season = None
    home_elo, away_elo, home_after, away_after = [], [], [], []

    for row in df.itertuples(index=False):
        if row.SeasonFile != season:
            season = row.SeasonFile
            ratings = {
                team: ELO_START + (1 - ELO_SEASON_REGRESSION) * (r - ELO_START)
                for team, r in ratings.items()
            }
        newcomer = ELO_START if season == first_season else ELO_NEW_TEAM
        home = ratings.get(row.HomeTeam, newcomer)
        away = ratings.get(row.AwayTeam, newcomer)
        home_elo.append(home)
        away_elo.append(away)

        if pd.isna(row.HomeGoals) or pd.isna(row.AwayGoals):
            home_after.append(home)
            away_after.append(away)
            continue
        expected = 1 / (1 + 10 ** ((away - home - ELO_HOME_ADVANTAGE) / 400))
        if row.HomeGoals == row.AwayGoals:
            actual = 0.5
        else:
            actual = 1.0 if row.HomeGoals > row.AwayGoals else 0.0
        margin = np.log1p(abs(row.HomeGoals - row.AwayGoals)) + 1
        change = ELO_K * margin * (actual - expected)
        ratings[row.HomeTeam] = home + change
        ratings[row.AwayTeam] = away - change
        home_after.append(home + change)
        away_after.append(away - change)

    df["HomeElo"] = home_elo
    df["AwayElo"] = away_elo
    df["HomeEloAfter"] = home_after
    df["AwayEloAfter"] = away_after
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["EloHomeWinProb"] = 1 / (1 + 10 ** (-(df["EloDiff"] + ELO_HOME_ADVANTAGE) / 400))
    return df


def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Each side's averages over its previous FORM_WINDOW league matches."""
    home_points = np.select(
        [df["HomeGoals"] > df["AwayGoals"], df["HomeGoals"] == df["AwayGoals"]], [3, 1], 0
    ).astype(float)
    away_points = np.select(
        [df["AwayGoals"] > df["HomeGoals"], df["AwayGoals"] == df["HomeGoals"]], [3, 1], 0
    ).astype(float)
    # A fixture not played yet earns no points, rather than counting as a defeat.
    unplayed = (df["HomeGoals"].isna() | df["AwayGoals"].isna()).to_numpy()
    home_points[unplayed] = np.nan
    away_points[unplayed] = np.nan

    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "Match": df.index,
                    "Side": "Home",
                    "Team": df["HomeTeam"],
                    "Date": df["Date"],
                    "Points": home_points,
                    "GoalsFor": df["HomeGoals"],
                    "GoalsAgainst": df["AwayGoals"],
                    "ShotsTargetFor": df["HomeShotsTarget"],
                    "ShotsTargetAgainst": df["AwayShotsTarget"],
                }
            ),
            pd.DataFrame(
                {
                    "Match": df.index,
                    "Side": "Away",
                    "Team": df["AwayTeam"],
                    "Date": df["Date"],
                    "Points": away_points,
                    "GoalsFor": df["AwayGoals"],
                    "GoalsAgainst": df["HomeGoals"],
                    "ShotsTargetFor": df["AwayShotsTarget"],
                    "ShotsTargetAgainst": df["HomeShotsTarget"],
                }
            ),
        ]
    ).sort_values(["Team", "Date", "Match"], kind="stable")

    by_team = long.groupby("Team", sort=False)
    for stat in ROLLING_STATS:
        # shift(1) keeps the current match out of its own features.
        long[f"{stat}L{FORM_WINDOW}"] = by_team[stat].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=2).mean()
        )
    long["RestDays"] = by_team["Date"].diff().dt.days.clip(upper=30)

    rolled = [f"{s}L{FORM_WINDOW}" for s in ROLLING_STATS] + ["RestDays"]
    for side in ("Home", "Away"):
        part = long[long["Side"] == side].set_index("Match")[rolled]
        df[[f"{side}{c}" for c in rolled]] = part.reindex(df.index).to_numpy()

    w = FORM_WINDOW
    df["FormPointsDiff"] = df[f"HomePointsL{w}"] - df[f"AwayPointsL{w}"]
    df["GoalDiffDiff"] = (df[f"HomeGoalsForL{w}"] - df[f"HomeGoalsAgainstL{w}"]) - (
        df[f"AwayGoalsForL{w}"] - df[f"AwayGoalsAgainstL{w}"]
    )
    df["ShotsTargetDiffDiff"] = (
        df[f"HomeShotsTargetForL{w}"] - df[f"HomeShotsTargetAgainstL{w}"]
    ) - (df[f"AwayShotsTargetForL{w}"] - df[f"AwayShotsTargetAgainstL{w}"])
    return df


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """All seasons in, one row per match out with every pre-match feature.

    Needs the full chronological history: Elo and form carry across seasons.
    """
    df = prepare_matches_dataframe(raw_df)
    df = add_elo(df)
    return add_rolling_form(df)
