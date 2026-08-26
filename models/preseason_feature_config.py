"""
Column definitions for the preseason (draft-time) model, shared by
train_preseason.py and predict_preseason.py.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "age",
    "is_rookie",
    "last_season_ppg",
    "two_season_avg_ppg",
    "games_played_last_season",
    "team_proj_ppg",
    "schedule_strength_rank",
]

CATEGORICAL_FEATURES = [
    "position",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fantasy_points_per_game"


def prepare_x(df):
    X = df[ALL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        # Guards against columns coming back as object dtype (e.g. from
        # a merge that produced missing values in some rows) - XGBoost
        # rejects object dtype outright, even when the values are really
        # just numbers-or-NaN. Coercing here is defensive and cheap.
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X
