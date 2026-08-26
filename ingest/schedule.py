"""
Pulls the season schedule from nflverse data and writes it into the
schedule table. Run once per season - the full schedule is known in
advance, so there's no need to re-run this weekly.

Usage:
    python ingest/schedule.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows


def fetch_schedule(season: int) -> list[dict]:
    print(f"Fetching schedule for {season}...")
    sched = nfl.load_schedules([season]).to_pandas()

    rows = []
    for _, r in sched.iterrows():
        rows.append({
            "season": season,
            "week": int(r["week"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "game_date": str(r.get("gameday")),
            "home_score": None if pd.isna(r.get("home_score")) else int(r["home_score"]),
            "away_score": None if pd.isna(r.get("away_score")) else int(r["away_score"]),
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = fetch_schedule(args.season)
    conn = get_connection()
    upsert_rows(conn, "schedule", rows)
    conn.close()
