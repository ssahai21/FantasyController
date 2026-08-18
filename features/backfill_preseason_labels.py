"""
After a season is complete, copies each player's actual average
fantasy points per game into preseason_features.fantasy_points_per_game
for that season - turning that season's preseason features into a
labeled training example for the preseason model.

Usage:
    python features/backfill_preseason_labels.py --season 2025
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection


def backfill(season: int) -> int:
    conn = get_connection()
    actuals = pd.read_sql(
        "SELECT player_id, AVG(fantasy_points) as ppg FROM player_stats "
        "WHERE season = ? GROUP BY player_id",
        conn, params=(season,),
    )

    if actuals.empty:
        print(f"No player_stats found for {season} - nothing to backfill.")
        conn.close()
        return 0

    cur = conn.cursor()
    updated = 0
    for _, r in actuals.iterrows():
        cur.execute(
            "UPDATE preseason_features SET fantasy_points_per_game = ? "
            "WHERE season = ? AND player_id = ?",
            (round(r["ppg"], 2), season, r["player_id"]),
        )
        updated += cur.rowcount

    conn.commit()
    conn.close()
    print(f"Backfilled {updated} preseason labels for {season}")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    backfill(args.season)
