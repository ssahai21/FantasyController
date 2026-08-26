"""
Pulls weekly player stats, snap counts, and target share from nflverse
data via the nfl_data_py package, and writes them into player_stats.

Usage:
    python ingest/stats.py --season 2026 --week 5
"""
import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

# Standard fantasy scoring weights (PPR). Adjust these to match your
# league's actual scoring settings once you wire in league_context data.
SCORING = {
    "pass_yards": 0.04,
    "pass_tds": 4,
    "interceptions": -2,
    "rush_yards": 0.1,
    "rush_tds": 6,
    "receptions": 1.0,       # PPR
    "rec_yards": 0.1,
    "rec_tds": 6,
    "fumbles_lost": -2,
}


def calculate_fantasy_points(row: pd.Series) -> float:
    points = 0.0
    points += row.get("passing_yards", 0) * SCORING["pass_yards"]
    points += row.get("passing_tds", 0) * SCORING["pass_tds"]
    points += row.get("interceptions", 0) * SCORING["interceptions"]
    points += row.get("rushing_yards", 0) * SCORING["rush_yards"]
    points += row.get("rushing_tds", 0) * SCORING["rush_tds"]
    points += row.get("receptions", 0) * SCORING["receptions"]
    points += row.get("receiving_yards", 0) * SCORING["rec_yards"]
    points += row.get("receiving_tds", 0) * SCORING["rec_tds"]
    points += row.get("rushing_fumbles_lost", 0) * SCORING["fumbles_lost"]
    points += row.get("receiving_fumbles_lost", 0) * SCORING["fumbles_lost"]
    return round(points, 2)


def fetch_week(season: int, week: int) -> list[dict]:
    print(f"Fetching weekly stats for {season} week {week}...")
    weekly = nfl.load_player_stats([season]).to_pandas()
    weekly = weekly[weekly["week"] == week]

    if weekly.empty:
        print(f"No data returned for {season} week {week} yet.")
        return []

    FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}
    weekly = weekly[weekly["position"].isin(FANTASY_POSITIONS)]

    # Snap counts and red zone touches come from a separate table
    try:
        snaps = nfl.load_snap_counts([season]).to_pandas()
        snaps = snaps[snaps["week"] == week][["pfr_player_id", "offense_pct"]]
    except Exception:
        snaps = pd.DataFrame(columns=["pfr_player_id", "offense_pct"])

    rows = []
    for _, r in weekly.iterrows():
        snap_pct = None
        if not snaps.empty:
            match = snaps[snaps["pfr_player_id"] == r.get("pfr_id")]
            if not match.empty:
                snap_pct = float(match.iloc[0]["offense_pct"])

        rows.append({
            "season": season,
            "week": week,
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
            "red_zone_touches": None,  # filled in once snap-count-style RZ data is added
            "air_yards": int(r.get("receiving_air_yards", 0) or 0),
            "fantasy_points": calculate_fantasy_points(r),
        })
    rows = [r for r in rows if pd.notna(r.get("player_id"))]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = fetch_week(args.season, args.week)
    conn = get_connection()
    upsert_rows(conn, "player_stats", rows)
    conn.close()