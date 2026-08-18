"""
Computes how hard each team's FULL upcoming season schedule is, by
position, before that season has been played. Since a team's own
current-season defense doesn't exist yet as data, this uses each
opponent's defensive performance from the PRIOR completed season as
the best available proxy for how tough they'll be to play against.

Usage:
    python features/build_schedule_strength.py --target-season 2026 --stats-season 2025
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows


def build_prior_season_def_rank(stats_season: int) -> pd.DataFrame:
    """Same idea as build_opponent_defense_rank in build_features.py,
    but over a FULL completed season rather than partial in-season data."""
    conn = get_connection()
    history = pd.read_sql(
        "SELECT opponent, position, fantasy_points FROM player_stats WHERE season = ?",
        conn, params=(stats_season,),
    )
    conn.close()

    allowed = (
        history.groupby(["opponent", "position"])["fantasy_points"]
        .mean()
        .reset_index()
        .rename(columns={"opponent": "team", "fantasy_points": "pts_allowed"})
    )
    allowed["opp_def_rank_allowed"] = (
        allowed.groupby("position")["pts_allowed"]
        .rank(method="min", ascending=True)
    )
    return allowed[["team", "position", "opp_def_rank_allowed"]]


def build(target_season: int, stats_season: int) -> list[dict]:
    conn = get_connection()
    schedule = pd.read_sql(
        "SELECT week, home_team, away_team FROM schedule WHERE season = ?",
        conn, params=(target_season,),
    )
    conn.close()

    if schedule.empty:
        print(f"No schedule found for {target_season}. Run ingest/schedule.py first.")
        return []

    def_rank = build_prior_season_def_rank(stats_season)
    if def_rank.empty:
        print(f"No player_stats found for {stats_season} to base defensive "
              f"ranks on. Need a completed prior season's data.")
        return []

    # Every team's full list of opponents this season
    home = schedule[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"})
    away = schedule[["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True)

    # Cross-join matchups with every position so we get a rank per
    # team/opponent/position, then average across all games per team/position
    positions = def_rank["position"].unique()
    matchups = matchups.merge(pd.Series(positions, name="position"), how="cross")

    joined = matchups.merge(
        def_rank.rename(columns={"team": "opponent"}),
        on=["opponent", "position"], how="left",
    )

    strength = joined.groupby(["team", "position"]).agg(
        avg_opp_def_rank=("opp_def_rank_allowed", "mean"),
    ).reset_index()
    strength["season"] = target_season
    strength = strength.round(2)

    return strength[["season", "team", "position", "avg_opp_def_rank"]].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-season", type=int, required=True,
                         help="The season whose schedule strength you're computing")
    parser.add_argument("--stats-season", type=int, required=True,
                         help="The prior completed season to base defensive ranks on")
    args = parser.parse_args()

    rows = build(args.target_season, args.stats_season)
    conn = get_connection()
    upsert_rows(conn, "schedule_strength", rows)
    conn.close()
