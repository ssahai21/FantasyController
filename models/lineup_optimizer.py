"""
Given your roster and this week's projections, finds the lineup that
maximizes total projected points, respecting your league's slot
requirements (QB, RB, WR, TE, FLEX, K, D/ST) and position eligibility.

Solved as an integer program rather than a heuristic: it's guaranteed
to find the mathematically best combination, not just a good-looking
one. With ~15-20 rostered players this solves in a fraction of a second.

Usage:
    python models/lineup_optimizer.py --season 2026 --week 6 --fantasy-team-id 3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pulp

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection

FLEX_ELIGIBLE = {"RB", "WR", "TE"}

# Slots that aren't "start a player to score points" slots and should
# be excluded from optimization
NON_STARTING_SLOTS = {"BENCH", "IR"}


def load_roster_with_projections(season: int, week: int, fantasy_team_id: str) -> pd.DataFrame:
    conn = get_connection()
    roster = pd.read_sql(
        "SELECT player_id FROM rosters WHERE season = ? AND week = ? AND fantasy_team_id = ?",
        conn, params=(season, week, str(fantasy_team_id)),
    )
    id_map = pd.read_sql("SELECT espn_player_id, nflverse_player_id FROM player_id_map", conn)
    projections = pd.read_sql(
        "SELECT player_id, player_name, position, predicted_points FROM projections "
        "WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    conn.close()

    roster = roster.merge(id_map, left_on="player_id", right_on="espn_player_id", how="left")
    roster = roster.merge(
        projections, left_on="nflverse_player_id", right_on="player_id",
        how="inner", suffixes=("", "_proj"),
    )
    return roster[["player_name", "position", "predicted_points"]]


def load_slots(season: int) -> dict:
    conn = get_connection()
    settings = pd.read_sql(
        "SELECT lineup_slot, slot_count FROM league_roster_settings WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()
    return {
        row["lineup_slot"]: int(row["slot_count"])
        for _, row in settings.iterrows()
        if row["lineup_slot"] not in NON_STARTING_SLOTS and row["slot_count"] > 0
    }


def slot_eligibility(position: str, slot: str) -> bool:
    if slot == "FLEX":
        return position in FLEX_ELIGIBLE
    return position == slot


def optimize(season: int, week: int, fantasy_team_id: str):
    roster = load_roster_with_projections(season, week, fantasy_team_id)
    if roster.empty:
        print("No roster + projections found for this team/week. Check that "
              "espn_league.py, predict_projection.py, and id_mapping.py have "
              "all been run for this season/week.")
        return

    slots = load_slots(season)
    if not slots:
        print("No league_roster_settings found. Run espn_league.py to pull them.")
        return

    players = roster.to_dict("records")
    prob = pulp.LpProblem("lineup_optimizer", pulp.LpMaximize)

    # One binary decision variable per (player, slot) pair where the
    # player is actually eligible for that slot
    x = {}
    for i, p in enumerate(players):
        for slot in slots:
            if slot_eligibility(p["position"], slot):
                x[(i, slot)] = pulp.LpVariable(f"x_{i}_{slot}", cat="Binary")

    # Objective: maximize total projected points across everyone assigned to a slot
    prob += pulp.lpSum(
        players[i]["predicted_points"] * x[(i, slot)] for (i, slot) in x
    )

    # Each player starts in at most one slot
    for i in range(len(players)):
        prob += pulp.lpSum(x[(i, slot)] for slot in slots if (i, slot) in x) <= 1

    # Each slot filled by exactly its required number of players
    # (or fewer, if you don't have enough eligible players - <= handles that)
    for slot, count in slots.items():
        prob += pulp.lpSum(x[(i, slot)] for i in range(len(players)) if (i, slot) in x) <= count

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    lineup = {slot: [] for slot in slots}
    started_indices = set()
    for (i, slot), var in x.items():
        if var.value() == 1:
            lineup[slot].append(players[i])
            started_indices.add(i)

    bench = [p for i, p in enumerate(players) if i not in started_indices]
    total_points = sum(p["predicted_points"] for slot_players in lineup.values() for p in slot_players)

    print(f"\nOptimal lineup for week {week} - projected total: {total_points:.2f} points\n")
    for slot, slot_players in lineup.items():
        for p in slot_players:
            print(f"  {slot:<6} {p['player_name']:<25} {p['predicted_points']:>6.2f} pts")

    if bench:
        print("\nBench:")
        for p in sorted(bench, key=lambda p: p["predicted_points"], reverse=True):
            print(f"  {'--':<6} {p['player_name']:<25} {p['predicted_points']:>6.2f} pts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--fantasy-team-id", required=True)
    args = parser.parse_args()

    optimize(args.season, args.week, args.fantasy_team_id)
