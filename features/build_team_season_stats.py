"""
Aggregates a completed season's schedule and player stats into
team-level offensive environment numbers - points scored/allowed per
game, plays per game, pass rate. This becomes the "team projection"
proxy for the FOLLOWING season, on the assumption that an offense's
recent baseline is a reasonable starting point before the new season's
own data exists.

Note: this is a simple heuristic, not a true team projection. It
doesn't know about coaching changes, key free agent departures, or
injuries - it's last year's reality, not next year's forecast. Good
enough as a baseline input feature; not a substitute for actual
analyst projections if you want to refine this later.

Usage:
    python features/build_team_season_stats.py --season 2025
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows


def build(season: int) -> list[dict]:
    conn = get_connection()
    games = pd.read_sql(
        "SELECT * FROM schedule WHERE season = ? AND home_score IS NOT NULL",
        conn, params=(season,),
    )
    plays = pd.read_sql(
        "SELECT team, week, pass_attempts, rush_attempts FROM player_stats WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if games.empty:
        print(f"No completed games found in schedule for {season}. "
              f"Run ingest/schedule.py (with scores) for a completed season.")
        return []

    # Points scored/allowed per game, from both home and away perspectives
    home = games[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "allowed"})
    away = games[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "allowed"})
    scoring = pd.concat([home, away], ignore_index=True)

    scoring_agg = scoring.groupby("team").agg(
        points_per_game=("scored", "mean"),
        points_allowed_per_game=("allowed", "mean"),
    ).reset_index()

    # Plays per game and pass rate, from team offensive stat totals
    plays_per_team_week = plays.groupby(["team", "week"]).agg(
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
    ).reset_index()
    plays_per_team_week["total_plays"] = (
        plays_per_team_week["pass_attempts"] + plays_per_team_week["rush_attempts"]
    )

    plays_agg = plays_per_team_week.groupby("team").agg(
        plays_per_game=("total_plays", "mean"),
        total_pass=("pass_attempts", "sum"),
        total_rush=("rush_attempts", "sum"),
    ).reset_index()
    plays_agg["pass_rate"] = plays_agg["total_pass"] / (
        plays_agg["total_pass"] + plays_agg["total_rush"]
    )

    merged = scoring_agg.merge(
        plays_agg[["team", "plays_per_game", "pass_rate"]], on="team", how="left"
    )
    merged["season"] = season
    merged = merged.round(2)

    cols = ["season", "team", "points_per_game", "points_allowed_per_game",
            "plays_per_game", "pass_rate"]
    return merged[cols].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = build(args.season)
    conn = get_connection()
    upsert_rows(conn, "team_season_stats", rows)
    conn.close()
