"""
Same purpose as models/feature_config.py, for the draft model instead
of the weekly one - one source of truth for train/predict scripts.
"""

NUMERIC_FEATURES = [
    "career_avg_ppg",
    "last_season_total_pts",
    "last_season_games",
    "team_proj_ppg",
    "schedule_strength",
]

CATEGORICAL_FEATURES = ["position"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "actual_season_pts"


def prepare_x(df):
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X
