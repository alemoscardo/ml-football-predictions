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
    "HC": "HomeCorners",
    "AC": "AwayCorners",
    "HF": "HomeFouls",
    "AF": "AwayFouls",
    "HY": "HomeYellows",
    "AY": "AwayYellows",
    "HR": "HomeReds",
    "AR": "AwayReds",
    "FTR": "Result",
}

LEGACY_FEATURES = [
    "HomeShots",
    "AwayShots",
    "HomeShotsTarget",
    "AwayShotsTarget",
    "HomeCorners",
    "AwayCorners",
    "HomeFouls",
    "AwayFouls",
    "HomeYellows",
    "AwayYellows",
    "HomeReds",
    "AwayReds",
]

ENGINEERED_FEATURES = [
    "ShotDiff",
    "ShotTargetDiff",
    "CornersDiff",
    "FoulsDiff",
    "YellowsDiff",
    "RedsDiff",
    "HomeShotAcc",
    "AwayShotAcc",
    "ShotAccDiff",
    "AggressionDiff",
]

ODDS_FEATURES = [
    "B365H",
    "B365D",
    "B365A",
    "NormProbHome_B365",
    "NormProbDraw_B365",
    "NormProbAway_B365",
    "OddsEdgeHome_B365",
]

MODEL_FEATURES = LEGACY_FEATURES + ENGINEERED_FEATURES + ODDS_FEATURES

REQUIRED_MATCH_STATS = LEGACY_FEATURES
OPTIONAL_ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def prepare_matches_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    rename_subset = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if rename_subset:
        df = df.rename(columns=rename_subset)

    df = _ensure_columns(df, REQUIRED_MATCH_STATS + OPTIONAL_ODDS_COLUMNS)

    df["ShotDiff"] = df["HomeShots"] - df["AwayShots"]
    df["ShotTargetDiff"] = df["HomeShotsTarget"] - df["AwayShotsTarget"]
    df["CornersDiff"] = df["HomeCorners"] - df["AwayCorners"]
    df["FoulsDiff"] = df["HomeFouls"] - df["AwayFouls"]
    df["YellowsDiff"] = df["HomeYellows"] - df["AwayYellows"]
    df["RedsDiff"] = df["HomeReds"] - df["AwayReds"]

    df["HomeShotAcc"] = df["HomeShotsTarget"] / df["HomeShots"].replace(0, np.nan)
    df["AwayShotAcc"] = df["AwayShotsTarget"] / df["AwayShots"].replace(0, np.nan)
    df["ShotAccDiff"] = df["HomeShotAcc"] - df["AwayShotAcc"]

    df["AggressionDiff"] = (
        df["HomeFouls"] + 2 * df["HomeYellows"] + 3 * df["HomeReds"]
    ) - (df["AwayFouls"] + 2 * df["AwayYellows"] + 3 * df["AwayReds"])

    df["ImpProbHome_B365"] = 1 / df["B365H"].replace(0, np.nan)
    df["ImpProbDraw_B365"] = 1 / df["B365D"].replace(0, np.nan)
    df["ImpProbAway_B365"] = 1 / df["B365A"].replace(0, np.nan)

    implied_sum = df[
        ["ImpProbHome_B365", "ImpProbDraw_B365", "ImpProbAway_B365"]
    ].sum(axis=1)
    df["NormProbHome_B365"] = df["ImpProbHome_B365"] / implied_sum
    df["NormProbDraw_B365"] = df["ImpProbDraw_B365"] / implied_sum
    df["NormProbAway_B365"] = df["ImpProbAway_B365"] / implied_sum

    df["OddsEdgeHome_B365"] = df["B365A"] - df["B365H"]

    return df


def build_feature_matrix(
    raw_df: pd.DataFrame,
    feature_list: list[str] | None = None,
) -> pd.DataFrame:
    features = MODEL_FEATURES if feature_list is None else feature_list
    prepared = prepare_matches_dataframe(raw_df)
    prepared = _ensure_columns(prepared, features)
    return prepared[features]
