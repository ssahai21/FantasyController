"""
For each team's full upcoming schedule, computes how tough it is per
position - the average defensive rank of the opponents they'll face,
based on last season's points-allowed-by-position (same defensive-rank
logic used weekly in build_features.py, applied at season scale and
projected onto next season's schedule).

Usage:
    python draft/schedule_strength.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows
from features.build_features import build_opponent_defense_rank


def compute_schedule_strength(season: int) -> list[dict]:
    prior_season = season - 1
    conn = get_connection()

    prior_stats = pd.read_sql(
        "SELECT * FROM player_stats WHERE season = ?", conn, params=(prior_season,))
    schedule = pd.read_sql(
        "SELECT * FROM schedule WHERE season = ?", conn, params=(season,))
    conn.close()

    if prior_stats.empty:
        print(f"No {prior_season} stats found - run historical_stats.py first.")
        return []
    if schedule.empty:
        print(f"No {season} schedule found - run ingest/schedule.py first.")
        return []

    def_rank = build_opponent_defense_rank(prior_stats)  # columns: team, position, opp_def_rank_allowed

    home = schedule[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"})
    away = schedule[["home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True)

    # Cross join matchups with every position so each team gets a
    # strength-of-schedule number per position, then bring in each
    # opponent's defensive rank for that position.
    positions = def_rank["position"].unique()
    matchups = matchups.merge(pd.DataFrame({"position": positions}), how="cross")
    matchups = matchups.merge(
        def_rank.rename(columns={"team": "opponent"}),
        on=["opponent", "position"], how="left",
    )

    strength = (
        matchups.groupby(["team", "position"])["opp_def_rank_allowed"]
        .mean()
        .reset_index()
        .rename(columns={"opp_def_rank_allowed": "avg_opp_def_rank"})
    )
    strength["season"] = season
    strength = strength.dropna(subset=["avg_opp_def_rank"])
    strength["avg_opp_def_rank"] = strength["avg_opp_def_rank"].round(1)

    return strength[["season", "team", "position", "avg_opp_def_rank"]].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = compute_schedule_strength(args.season)
    conn = get_connection()
    upsert_rows(conn, "schedule_strength", rows)
    conn.close()
