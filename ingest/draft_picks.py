"""
Pulls NFL draft picks (round, overall pick) via nfl_data_py. Draft
capital is the single strongest predictor of rookie fantasy production
available before a rookie has played a game - teams invest draft
capital based on talent evaluation most fans and even most models
can't replicate from box scores alone.

Usage:
    python ingest/draft_picks.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}


def fetch(draft_year: int) -> list[dict]:
    print(f"Fetching draft picks for {draft_year}...")
    picks = nfl.load_draft_picks([draft_year]).to_pandas()
    picks = picks[picks["position"].isin(FANTASY_POSITIONS)]

    if picks.empty:
        print(f"No draft picks found for {draft_year}.")
        return []

    rows = []
    for _, r in picks.iterrows():
        rows.append({
            "draft_year": draft_year,
            "player_id": r.get("gsis_id"),
            "player_name": r.get("pfr_player_name") or r.get("cfb_player_name"),
            "position": r.get("position"),
            "team": r.get("team"),
            "round": int(r["round"]) if not pd.isna(r.get("round")) else None,
            "overall_pick": int(r["pick"]) if not pd.isna(r.get("pick")) else None,
        })

    # Some rookies don't get a gsis_id assigned until they're on an NFL
    # roster - these can't be joined to player_stats later and are
    # dropped rather than stored with a useless null ID
    rows = [r for r in rows if pd.notna(r.get("player_id"))]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                         help="The draft year to pull")
    args = parser.parse_args()

    rows = fetch(args.season)
    conn = get_connection()
    upsert_rows(conn, "draft_capital", rows)
    conn.close()
