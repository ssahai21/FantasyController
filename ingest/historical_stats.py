"""
Bulk-pulls multiple past seasons of weekly player stats in one call -
unlike stats.py, which pulls one current week at a time, this is meant
to be run once to backfill history that draft features need.

Usage:
    python ingest/historical_stats.py --seasons 2022 2023 2024 2025
"""
import argparse
import sys
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows
from ingest.stats import calculate_fantasy_points  # reuse, don't duplicate scoring logic


def fetch_seasons(seasons: list[int]) -> list[dict]:
    print(f"Fetching weekly stats for seasons {seasons} (this pulls full "
          f"seasons, may take a moment)...")
    weekly = nfl.import_weekly_data(seasons)
    weekly = weekly[weekly["season_type"] == "REG"]  # regular season only

    try:
        snaps = nfl.import_snap_counts(seasons)[["pfr_player_id", "season", "week", "offense_pct"]]
    except Exception:
        snaps = pd.DataFrame(columns=["pfr_player_id", "season", "week", "offense_pct"])

    rows = []
    for _, r in weekly.iterrows():
        snap_pct = None
        if not snaps.empty:
            match = snaps[
                (snaps["pfr_player_id"] == r.get("pfr_id"))
                & (snaps["season"] == r["season"])
                & (snaps["week"] == r["week"])
            ]
            if not match.empty:
                snap_pct = float(match.iloc[0]["offense_pct"])

        rows.append({
            "season": int(r["season"]),
            "week": int(r["week"]),
            "player_id": r["player_id"],
            "player_name": r["player_display_name"],
            "position": r.get("position"),
            "team": r.get("recent_team"),
            "opponent": r.get("opponent_team"),
            "snap_pct": snap_pct,
            "targets": int(r.get("targets", 0) or 0),
            "receptions": int(r.get("receptions", 0) or 0),
            "rec_yards": int(r.get("receiving_yards", 0) or 0),
            "rush_attempts": int(r.get("carries", 0) or 0),
            "rush_yards": int(r.get("rushing_yards", 0) or 0),
            "pass_attempts": int(r.get("attempts", 0) or 0),
            "pass_yards": int(r.get("passing_yards", 0) or 0),
            "pass_tds": int(r.get("passing_tds", 0) or 0),
            "rush_tds": int(r.get("rushing_tds", 0) or 0),
            "rec_tds": int(r.get("receiving_tds", 0) or 0),
            "red_zone_touches": None,
            "air_yards": int(r.get("receiving_air_yards", 0) or 0),
            "fantasy_points": calculate_fantasy_points(r),
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()

    rows = fetch_seasons(args.seasons)
    conn = get_connection()
    upsert_rows(conn, "player_stats", rows)
    conn.close()
