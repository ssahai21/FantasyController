"""
Builds draft-time features for a target season: each player's recent
career performance, their team's offensive environment (proxied from
the prior season), and their full-season schedule strength - combined
into one row per player, usable before a single game of the target
season is played.

Run build_team_season_stats.py and build_schedule_strength.py for the
relevant prior season(s) before this.

Usage:
    python features/build_preseason_features.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

# How many prior seasons of player performance to consider, and how to
# weight them - most recent season counts more than the one before it.
SEASON_WEIGHTS = {0: 0.65, 1: 0.35}  # 0 = most recent prior season


def player_season_averages(season: int) -> pd.DataFrame:
    """One row per player for a given completed season: games played
    and average fantasy points per game that season."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT player_id, player_name, position, team, week, fantasy_points "
        "FROM player_stats WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "position",
                                       "team", "games_played", "ppg"])

    agg = df.groupby("player_id").agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        games_played=("week", "nunique"),
        ppg=("fantasy_points", "mean"),
    ).reset_index()
    return agg


def build(season: int) -> list[dict]:
    prior_seasons = [season - 1, season - 2]
    season_data = {i: player_season_averages(s) for i, s in enumerate(prior_seasons)}

    if season_data[0].empty:
        print(f"No player_stats found for {prior_seasons[0]} (most recent prior "
              f"season). Need at least one completed prior season to build "
              f"preseason features.")
        return []

    # Start from the most recent prior season's player list - that's
    # who's realistically draftable
    base = season_data[0][["player_id", "player_name", "position", "team",
                             "games_played", "ppg"]].rename(
        columns={"games_played": "games_played_last_season", "ppg": "last_season_ppg"})

    # Weighted multi-season average, falling back to just last season's
    # number for players without two full prior seasons of data
    weighted = season_data[0][["player_id", "ppg"]].rename(columns={"ppg": "ppg_0"})
    if not season_data[1].empty:
        weighted = weighted.merge(
            season_data[1][["player_id", "ppg"]].rename(columns={"ppg": "ppg_1"}),
            on="player_id", how="left",
        )
    else:
        weighted["ppg_1"] = None

    weighted["two_season_avg_ppg"] = weighted.apply(
        lambda r: (
            r["ppg_0"] * SEASON_WEIGHTS[0] + r["ppg_1"] * SEASON_WEIGHTS[1]
            if pd.notnull(r["ppg_1"])
            else r["ppg_0"]  # no second season available, just use the one we have
        ),
        axis=1,
    )

    df = base.merge(weighted[["player_id", "two_season_avg_ppg"]], on="player_id", how="left")

    # Team environment proxy, from the most recent prior season
    conn = get_connection()
    team_stats = pd.read_sql(
        "SELECT team, points_per_game FROM team_season_stats WHERE season = ?",
        conn, params=(prior_seasons[0],),
    )
    sched_strength = pd.read_sql(
        "SELECT team, position, avg_opp_def_rank FROM schedule_strength WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    df = df.merge(
        team_stats.rename(columns={"points_per_game": "team_proj_ppg"}),
        on="team", how="left",
    )
    df = df.merge(sched_strength.rename(columns={"avg_opp_def_rank": "schedule_strength_rank"}),
                   on=["team", "position"], how="left")

    df["season"] = season
    df["fantasy_points_per_game"] = None  # filled in after the season, for training

    cols = ["season", "player_id", "player_name", "position", "team",
            "last_season_ppg", "two_season_avg_ppg", "games_played_last_season",
            "team_proj_ppg", "schedule_strength_rank", "fantasy_points_per_game"]
    df = df[cols].round(2).where(pd.notnull(df), None)
    return df.to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = build(args.season)
    conn = get_connection()
    upsert_rows(conn, "preseason_features", rows)
    conn.close()
