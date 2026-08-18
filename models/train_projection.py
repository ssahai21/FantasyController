"""
Trains an XGBoost regression model to predict weekly fantasy points,
using every labeled row (fantasy_points IS NOT NULL, via
backfill_labels.py) currently in the features table.

Validates with a TIME-BASED split: trains on all weeks except the most
recent one, tests on that most recent week. A random shuffle split
would let the model "see" similar weeks on both sides and look better
than it actually is - a time split mimics how it's really used
(predicting a future week from past ones).

Usage:
    python models/train_projection.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection
from models.feature_config import ALL_FEATURES, TARGET, prepare_x

MODEL_PATH = Path(__file__).parent / "artifacts" / "projection_model.json"


def load_labeled_data(season: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT * FROM features WHERE season = ? AND {TARGET} IS NOT NULL",
        conn, params=(season,),
    )
    conn.close()
    return df


def train(season: int):
    df = load_labeled_data(season)
    if df.empty:
        print("No labeled data yet. Run backfill_labels.py for at least "
              "a couple of played weeks first.")
        return

    weeks = sorted(df["week"].unique())
    if len(weeks) < 2:
        print(f"Only {len(weeks)} labeled week(s) available - need at least "
              f"2 (one to train on, one to validate against). Backfill more "
              f"weeks and try again.")
        return

    test_week = weeks[-1]
    train_df = df[df["week"] != test_week]
    test_df = df[df["week"] == test_week]

    X_train, y_train = prepare_x(train_df), train_df[TARGET].astype(float)
    X_test, y_test = prepare_x(test_df), test_df[TARGET].astype(float)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # A naive "predict the position average" baseline - if the model
    # doesn't beat this, it isn't earning its complexity.
    position_avg = train_df.groupby("position")[TARGET].mean()
    baseline_preds = test_df["position"].map(position_avg).fillna(y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    print(f"\nValidated on week {test_week} ({len(test_df)} players, "
          f"trained on weeks {weeks[:-1]}):")
    print(f"  Model MAE:     {mae:.2f} fantasy points")
    print(f"  Model RMSE:    {rmse:.2f} fantasy points")
    print(f"  Baseline MAE:  {baseline_mae:.2f} fantasy points "
          f"(position-average, no model)")
    if mae < baseline_mae:
        print(f"  Model beats the naive baseline by {baseline_mae - mae:.2f} points/player.")
    else:
        print("  WARNING: model is not beating the naive baseline. With this "
              "little data that's expected early in a season - keep collecting "
              "weeks before trusting projections over your gut.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    importances = pd.Series(model.feature_importances_, index=ALL_FEATURES)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    train(args.season)
