"""
Single source of truth for which columns from `features` feed the
projection model. Both train_projection.py and predict_projection.py
import this - if they ever used different column lists, predictions
would silently be wrong.
"""

NUMERIC_FEATURES = [
    "avg_pts_l3",
    "avg_pts_season",
    "target_share_trend",
    "snap_pct_l3",
    "opp_def_rank_allowed",
    "implied_team_total",
]

CATEGORICAL_FEATURES = [
    "position",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fantasy_points"


def prepare_x(df):
    """Selects feature columns and marks categoricals so XGBoost's
    native categorical support can use them without manual one-hot
    encoding."""
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X
