"""
Builds draft_features: one row per player for `season`, combining
recency-weighted career history with that season's team and schedule
outlook. Can be run for a past season (to generate training data, if
that season's results are already in player_stats) or an upcoming
season (to generate real draft projections - actual_season_pts stays
NULL until that season is played).

Usage:
    python draft/build_draft_features.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

# Recency weights for up to 3 prior seasons: most recent counts most.
# Renormalized automatically if a player has fewer than 3 prior seasons.
SEASON_WEIGHTS = [0.5, 0.3, 0.2]


def build_draft_features(season: int) -> list[dict]:
    conn = get_connection()

    prior_seasons = [season - i for i in range(1, len(SEASON_WEIGHTS) + 1)]
    history = pd.read_sql(
        f"SELECT * FROM player_stats WHERE season IN "
        f"({','.join('?' * len(prior_seasons))})",
        conn, params=prior_seasons,
    )
    team_proj = pd.read_sql(
        "SELECT team, proj_ppg FROM team_projections WHERE season = ?",
        conn, params=(season,),
    )
    sched_strength = pd.read_sql(
        "SELECT team, position, avg_opp_def_rank FROM schedule_strength WHERE season = ?",
        conn, params=(season,),
    )
    actuals = pd.read_sql(
        "SELECT player_id, SUM(fantasy_points) as actual_season_pts "
        "FROM player_stats WHERE season = ? GROUP BY player_id",
        conn, params=(season,),
    )
    conn.close()

    if history.empty:
        print(f"No history found for prior seasons {prior_seasons}. "
              f"Run historical_stats.py first.")
        return []

    # Per-player, per-season totals and games played
    season_totals = (
        history.groupby(["season", "player_id"])
        .agg(player_name=("player_name", "last"),
             position=("position", "last"),
             team=("team", "last"),
             season_pts=("fantasy_points", "sum"),
             games=("fantasy_points", "count"))
        .reset_index()
    )
    season_totals["ppg"] = season_totals["season_pts"] / season_totals["games"].clip(lower=1)

    rows = []
    for player_id, group in season_totals.groupby("player_id"):
        group = group.set_index("season")
        weighted_sum, weight_total = 0.0, 0.0
        for offset, weight in enumerate(SEASON_WEIGHTS, start=1):
            s = season - offset
            if s in group.index:
                weighted_sum += group.loc[s, "ppg"] * weight
                weight_total += weight
        if weight_total == 0:
            continue  # no usable history for this player at all
        career_avg_ppg = round(weighted_sum / weight_total, 2)

        most_recent = group.loc[group.index.max()]
        last_season = season - 1
        last_season_row = group.loc[last_season] if last_season in group.index else None

        rows.append({
            "season": season,
            "player_id": player_id,
            "player_name": most_recent["player_name"],
            "position": most_recent["position"],
            "team": most_recent["team"],
            "career_avg_ppg": career_avg_ppg,
            "last_season_total_pts": round(float(last_season_row["season_pts"]), 2) if last_season_row is not None else None,
            "last_season_games": int(last_season_row["games"]) if last_season_row is not None else 0,
        })

    df = pd.DataFrame(rows)
    df = df.merge(team_proj, on="team", how="left").rename(columns={"proj_ppg": "team_proj_ppg"})
    df = df.merge(sched_strength, on=["team", "position"], how="left").rename(
        columns={"avg_opp_def_rank": "schedule_strength"})
    df = df.merge(actuals, on="player_id", how="left")

    df["season"] = season
    columns = [
        "season", "player_id", "player_name", "position", "team",
        "career_avg_ppg", "last_season_total_pts", "last_season_games",
        "team_proj_ppg", "schedule_strength", "actual_season_pts",
    ]
    df = df[columns].where(pd.notnull(df), None)
    return df.to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = build_draft_features(args.season)
    conn = get_connection()
    upsert_rows(conn, "draft_features", rows)
    conn.close()
