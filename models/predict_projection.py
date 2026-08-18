"""
Loads the trained projection model and generates predicted fantasy
points for every player with features built for the target week,
writing results into the projections table.

Usage:
    python models/predict_projection.py --season 2026 --week 6
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows
from models.feature_config import prepare_x

MODEL_PATH = Path(__file__).parent / "artifacts" / "projection_model.json"


def predict(season: int, week: int) -> list[dict]:
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}. Run "
              f"train_projection.py first.")
        return []

    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM features WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    conn.close()

    if df.empty:
        print(f"No features found for {season} week {week}. "
              f"Run build_features.py for this week first.")
        return []

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    X = prepare_x(df)
    df["predicted_points"] = model.predict(X).round(2)

    cols = ["season", "week", "player_id", "player_name", "position",
            "team", "predicted_points"]
    return df[cols].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = predict(args.season, args.week)
    if rows:
        conn = get_connection()
        upsert_rows(conn, "projections", rows)
        conn.close()
        top10 = sorted(rows, key=lambda r: r["predicted_points"], reverse=True)[:10]
        print("\nTop 10 projections:")
        for r in top10:
            print(f"  {r['player_name']:<25} {r['position']:<4} "
                  f"{r['predicted_points']:>6.2f} pts")
