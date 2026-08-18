"""
Trains an XGBoost model to predict a player's season-long average
fantasy points per game, using only information available BEFORE that
season started (career performance, team environment, schedule
strength). Needs preseason_features built and labeled (via
backfill_preseason_labels.py) for at least two past seasons.

Validates with a season-based split: trains on all seasons except the
most recent one, tests on that one - same "don't let the model see the
future" logic as the weekly model's week-based split.

Usage:
    python models/train_preseason.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection
from models.preseason_feature_config import ALL_FEATURES, TARGET, prepare_x

MODEL_PATH = Path(__file__).parent / "artifacts" / "preseason_model.json"


def load_labeled_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT * FROM preseason_features WHERE {TARGET} IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def train():
    df = load_labeled_data()
    if df.empty:
        print("No labeled preseason data yet. Build and backfill "
              "preseason_features for at least 2 past seasons first.")
        return

    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print(f"Only {len(seasons)} labeled season(s) available - need at "
              f"least 2. Run build_preseason_features.py + "
              f"backfill_preseason_labels.py for another past season.")
        return

    test_season = seasons[-1]
    train_df = df[df["season"] != test_season]
    test_df = df[df["season"] == test_season]

    X_train, y_train = prepare_x(train_df), train_df[TARGET].astype(float)
    X_test, y_test = prepare_x(test_df), test_df[TARGET].astype(float)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,          # shallower than the weekly model - far less
                               # training data here, so less room for depth
        learning_rate=0.05,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    baseline_preds = train_df.groupby("position")[TARGET].mean()
    baseline = test_df["position"].map(baseline_preds).fillna(y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline)

    print(f"\nValidated on {test_season} season ({len(test_df)} players, "
          f"trained on seasons {seasons[:-1]}):")
    print(f"  Model MAE:    {mae:.2f} pts/game")
    print(f"  Model RMSE:   {rmse:.2f} pts/game")
    print(f"  Baseline MAE: {baseline_mae:.2f} pts/game (position average)")
    if mae >= baseline_mae:
        print("  WARNING: not beating the naive baseline. With only a "
              "couple of training seasons this is common - the model "
              "needs several past seasons before it reliably adds value "
              "over simple heuristics.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    importances = pd.Series(model.feature_importances_, index=ALL_FEATURES)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    train()
