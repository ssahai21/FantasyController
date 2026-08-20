"""
Pulls weekly injury and practice reports via nfl_data_py and writes
them into injury_reports.

Usage:
    python ingest/injuries.py --season 2026 --week 5
"""
import argparse
import sys
import urllib.error
from pathlib import Path

import nfl_data_py as nfl

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows


def fetch_week(season: int, week: int) -> list[dict]:
    print(f"Fetching injury reports for {season} week {week}...")
    try:
        injuries = nfl.import_injuries([season])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"No injury data file published yet for {season} - this is "
                  f"normal before the season starts. Nothing to do.")
            return []
        raise

    injuries = injuries[injuries["week"] == week]

    if injuries.empty:
        print(f"No injury data returned for {season} week {week} yet.")
        return []

    rows = []
    for _, r in injuries.iterrows():
        rows.append({
            "season": season,
            "week": week,
            "player_id": r["gsis_id"],
            "player_name": r["full_name"],
            "team": r.get("team"),
            "report_status": r.get("report_status"),
            "practice_status": r.get("practice_status"),
            "injury_description": r.get("report_primary_injury"),
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = fetch_week(args.season, args.week)
    conn = get_connection()
    upsert_rows(conn, "injury_reports", rows)
    conn.close()
