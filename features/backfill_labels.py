"""
After a week has been played and stats.py has pulled the real results,
this copies actual fantasy_points from player_stats into the matching
row in features - turning that row from a prediction target into a
labeled training example.

Run this once a week's games are done, before training or retraining
the projection model.

Usage:
    python features/backfill_labels.py --season 2026 --week 5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection


def backfill(season: int, week: int) -> int:
    conn = get_connection()

    results = pd.read_sql(
        "SELECT player_id, fantasy_points FROM player_stats WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    if results.empty:
        print(f"No results in player_stats for {season} week {week} yet - "
              f"run ingest/stats.py for this week first.")
        conn.close()
        return 0

    cur = conn.cursor()
    updated = 0
    for _, r in results.iterrows():
        cur.execute(
            "UPDATE features SET fantasy_points = ? "
            "WHERE season = ? AND week = ? AND player_id = ?",
            (r["fantasy_points"], season, week, r["player_id"]),
        )
        updated += cur.rowcount

    conn.commit()
    conn.close()
    print(f"Backfilled {updated} labels for {season} week {week}")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    backfill(args.season, args.week)
