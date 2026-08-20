"""
Loads the trained preseason model and generates predicted season-long
average fantasy points per game for every player with preseason
features built for the target season. This is what draft rankings
should be based on.

Usage:
    python models/predict_preseason.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows
from models.preseason_feature_config import prepare_x

MODEL_PATH = Path(__file__).parent / "artifacts" / "preseason_model.json"


def predict(season: int) -> list[dict]:
    if not MODEL_PATH.exists():
        print(f"No trained preseason model at {MODEL_PATH}. Run "
              f"train_preseason.py first.")
        return []

    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM preseason_features WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if df.empty:
        print(f"No preseason_features found for {season}. Run "
              f"build_preseason_features.py for this season first.")
        return []

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    df['team_proj_ppg'] = pd.to_numeric(df['team_proj_ppg'], errors='coerce')
    X = prepare_x(df)
    df["predicted_ppg"] = model.predict(X).round(2)

    cols = ["season", "player_id", "player_name", "position", "team", "predicted_ppg"]
    return df[cols].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = predict(args.season)
    if rows:
        conn = get_connection()
        upsert_rows(conn, "preseason_projections", rows)
        conn.close()
        top15 = sorted(rows, key=lambda r: r["predicted_ppg"], reverse=True)[:15]
        print("\nTop 15 draft projections:")
        for r in top15:
            print(f"  {r['player_name']:<25} {r['position']:<4} "
                  f"{r['predicted_ppg']:>6.2f} pts/gm")
