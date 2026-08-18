"""
Computes a need score per position for your team: positive means your
current starters at that position are weaker than a typical rostered
player league-wide (a weakness worth addressing), negative means
you're already strong there.

Works in two modes:
  - In-season (--week N): uses actual roster + weekly projections
  - Draft/preseason (--week 0): uses whatever roster exists so far
    (empty before the draft starts) + preseason projections

Usage:
    python features/build_team_needs.py --season 2026 --week 5 --fantasy-team-id 3
    python features/build_team_needs.py --season 2026 --week 0 --fantasy-team-id 3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

# FLEX-eligible positions - a league's FLEX slot count gets split across
# these when figuring out how many "starters" you need per position.
# This is a simplification: real FLEX allocation depends on your actual
# lineup choices, not a fixed split.
FLEX_POSITIONS = ["RB", "WR", "TE"]


def load_projections(season: int, week: int) -> pd.DataFrame:
    conn = get_connection()
    if week == 0:
        df = pd.read_sql(
            "SELECT player_id, player_name, position, predicted_ppg AS value "
            "FROM preseason_projections WHERE season = ?",
            conn, params=(season,),
        )
    else:
        df = pd.read_sql(
            "SELECT player_id, player_name, position, predicted_points AS value "
            "FROM projections WHERE season = ? AND week = ?",
            conn, params=(season, week),
        )
    conn.close()
    return df


def starter_slots_by_position(season: int) -> dict:
    conn = get_connection()
    settings = pd.read_sql(
        "SELECT lineup_slot, slot_count FROM league_roster_settings WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    slots = {row["lineup_slot"]: int(row["slot_count"]) for _, row in settings.iterrows()}
    flex_count = slots.get("FLEX", 0)

    # Split FLEX evenly across RB/WR/TE as a simplification
    result = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "D/ST"]:
        result[pos] = slots.get(pos, 0)
    if flex_count:
        share = flex_count / len(FLEX_POSITIONS)
        for pos in FLEX_POSITIONS:
            result[pos] = result.get(pos, 0) + share

    return result


def build(season: int, week: int, fantasy_team_id: str) -> list[dict]:
    projections = load_projections(season, week)
    if projections.empty:
        print(f"No projections found for season {season} week {week}. "
              f"Run predict_projection.py or predict_preseason.py first.")
        return []

    conn = get_connection()
    roster_week = week if week > 0 else pd.read_sql(
        "SELECT MAX(week) as w FROM rosters WHERE season = ?", conn, params=(season,)
    )["w"].iloc[0] or 0
    all_rosters = pd.read_sql(
        "SELECT fantasy_team_id, player_id FROM rosters WHERE season = ? AND week = ?",
        conn, params=(season, roster_week),
    )
    id_map = pd.read_sql(
        "SELECT espn_player_id, nflverse_player_id FROM player_id_map", conn
    )
    conn.close()

    # Bridge ESPN roster IDs to nflverse IDs so we can join to projections
    all_rosters = all_rosters.merge(
        id_map, left_on="player_id", right_on="espn_player_id", how="left"
    )
    all_rosters = all_rosters.merge(
        projections, left_on="nflverse_player_id", right_on="player_id",
        how="inner", suffixes=("", "_proj"),
    )

    slots = starter_slots_by_position(season)
    your_roster = all_rosters[all_rosters["fantasy_team_id"] == str(fantasy_team_id)]

    rows = []
    for position, starter_count in slots.items():
        if starter_count <= 0:
            continue

        # Replacement baseline: average value of ALL rostered players
        # league-wide at this position - a proxy for "typical startable"
        league_pool = all_rosters[all_rosters["position"] == position]
        replacement_baseline = league_pool["value"].mean() if not league_pool.empty else None

        # Your starters: your top N rostered players at this position,
        # N = how many starting slots your league requires
        your_pool = your_roster[your_roster["position"] == position].sort_values(
            "value", ascending=False)
        top_n = your_pool.head(round(starter_count))
        your_starter_avg = top_n["value"].mean() if not top_n.empty else 0.0

        if replacement_baseline is None:
            continue

        need_score = round(replacement_baseline - your_starter_avg, 2)
        rows.append({
            "season": season,
            "week": week,
            "fantasy_team_id": str(fantasy_team_id),
            "position": position,
            "your_starter_avg": round(your_starter_avg, 2),
            "replacement_baseline": round(replacement_baseline, 2),
            "need_score": need_score,
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True,
                         help="Use 0 for draft/preseason needs")
    parser.add_argument("--fantasy-team-id", required=True)
    args = parser.parse_args()

    rows = build(args.season, args.week, args.fantasy_team_id)
    conn = get_connection()
    upsert_rows(conn, "team_needs", rows)
    conn.close()

    print("\nTeam needs (positive = weakness, negative = strength):")
    for r in sorted(rows, key=lambda r: r["need_score"], reverse=True):
        print(f"  {r['position']:<6} need_score={r['need_score']:>6.2f}  "
              f"(you: {r['your_starter_avg']:.2f}, league: {r['replacement_baseline']:.2f})")
