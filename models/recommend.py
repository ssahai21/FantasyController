"""
Ranks available players by combining projected value with how much you
need that position, so a merely-good player at a position you're weak
in can outrank a slightly-better player at a position you've already
got covered.

final_score = projected_value * (1 + NEED_WEIGHT * normalized_need)

Modes (--pool):
  draft    - all players not yet on ANY roster (nobody's drafted them)
  waivers  - same as draft, for in-season free agent pickups
  trade:<other_team_id> - a specific opponent's roster, to find who
             would help you most if you traded for them

Usage:
    python models/recommend.py --season 2026 --week 0 --fantasy-team-id 3 --pool draft
    python models/recommend.py --season 2026 --week 6 --fantasy-team-id 3 --pool waivers
    python models/recommend.py --season 2026 --week 6 --fantasy-team-id 3 --pool trade:7
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection

# How strongly need shifts the ranking. 0 = ignore need entirely (pure
# best-player-available). Higher values push need-filling harder.
# 0.4 means a maximal-need position gets up to a 40% score boost.
NEED_WEIGHT = 0.4


def load_projections(season: int, week: int) -> pd.DataFrame:
    conn = get_connection()
    if week == 0:
        df = pd.read_sql(
            "SELECT player_id, player_name, position, team, predicted_ppg AS value "
            "FROM preseason_projections WHERE season = ?",
            conn, params=(season,),
        )
        conn.close()
        df["play_probability"] = 1.0  # no in-season injury data exists at draft time
    else:
        df = pd.read_sql(
            "SELECT player_id, player_name, position, team, predicted_points AS value "
            "FROM projections WHERE season = ? AND week = ?",
            conn, params=(season, week),
        )
        risk = pd.read_sql(
            "SELECT player_id, play_probability FROM injury_risk WHERE season = ? AND week = ?",
            conn, params=(season, week),
        )
        conn.close()
        df = df.merge(risk, on="player_id", how="left")
        df["play_probability"] = df["play_probability"].fillna(1.0)

    df["value"] = df["value"] * df["play_probability"]
    return df


def load_rosters(season: int, week: int) -> pd.DataFrame:
    conn = get_connection()
    roster_week = week if week > 0 else pd.read_sql(
        "SELECT MAX(week) as w FROM rosters WHERE season = ?", conn, params=(season,)
    )["w"].iloc[0] or 0
    rosters = pd.read_sql(
        "SELECT fantasy_team_id, player_id FROM rosters WHERE season = ? AND week = ?",
        conn, params=(season, roster_week),
    )
    id_map = pd.read_sql("SELECT espn_player_id, nflverse_player_id FROM player_id_map", conn)
    conn.close()

    rosters = rosters.merge(id_map, left_on="player_id", right_on="espn_player_id", how="left")
    return rosters[["fantasy_team_id", "nflverse_player_id"]].rename(
        columns={"nflverse_player_id": "player_id"})


def load_needs(season: int, week: int, fantasy_team_id: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT position, need_score FROM team_needs "
        "WHERE season = ? AND week = ? AND fantasy_team_id = ?",
        conn, params=(season, week, str(fantasy_team_id)),
    )
    conn.close()
    return df


def normalize_needs(needs: pd.DataFrame) -> dict:
    """Scales need scores to roughly [-1, 1] so NEED_WEIGHT has a
    consistent, predictable effect regardless of the raw point scale."""
    if needs.empty:
        return {}
    max_abs = needs["need_score"].abs().max()
    if max_abs == 0:
        return {row["position"]: 0.0 for _, row in needs.iterrows()}
    return {row["position"]: row["need_score"] / max_abs for _, row in needs.iterrows()}


def recommend(season: int, week: int, fantasy_team_id: str, pool: str, top_n: int = 15) -> pd.DataFrame:
    projections = load_projections(season, week)
    if projections.empty:
        print("No projections available for this season/week.")
        return pd.DataFrame()

    rosters = load_rosters(season, week)
    rostered_ids = set(rosters["player_id"].dropna())

    if pool in ("draft", "waivers"):
        available = projections[~projections["player_id"].isin(rostered_ids)]
    elif pool.startswith("trade:"):
        other_team_id = pool.split(":", 1)[1]
        target_ids = set(rosters[rosters["fantasy_team_id"] == other_team_id]["player_id"])
        available = projections[projections["player_id"].isin(target_ids)]
    else:
        raise ValueError(f"Unknown pool: {pool}. Use draft, waivers, or trade:<team_id>")

    if available.empty:
        print("No players found in this pool - check the team ID or that "
              "rosters/projections are populated.")
        return pd.DataFrame()

    needs = load_needs(season, week, fantasy_team_id)
    needs_map = normalize_needs(needs)

    available = available.copy()
    available["need_multiplier"] = available["position"].map(needs_map).fillna(0.0)
    available["final_score"] = available["value"] * (1 + NEED_WEIGHT * available["need_multiplier"])
    available = available.sort_values("final_score", ascending=False)

    return available.head(top_n)[
        ["player_name", "position", "team", "value", "need_multiplier", "final_score"]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True, help="0 for draft mode")
    parser.add_argument("--fantasy-team-id", required=True)
    parser.add_argument("--pool", required=True,
                         help="'draft', 'waivers', or 'trade:<other_team_id>'")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    results = recommend(args.season, args.week, args.fantasy_team_id, args.pool, args.top_n)
    if not results.empty:
        print(f"\nTop {len(results)} recommendations ({args.pool}):")
        print(f"{'Player':<25}{'Pos':<6}{'Team':<6}{'Value':>8}{'Need x':>9}{'Score':>9}")
        for _, r in results.iterrows():
            print(f"{r['player_name']:<25}{r['position']:<6}{r['team']:<6}"
                  f"{r['value']:>8.2f}{r['need_multiplier']:>9.2f}{r['final_score']:>9.2f}")
