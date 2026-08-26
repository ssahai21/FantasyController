"""
Projects each team's points-per-game for `season` using their actual
scoring from `season - 1`, shrunk toward the league average. Shrinkage
matters because a team that scored 31 ppg last year on some late-season
garbage-time luck isn't necessarily a true-talent 31 ppg team - blending
with the league mean tempers small-sample overreaction.

This is a simple baseline. A better version would blend in Vegas
season win-total odds where available - not wired up yet, see README.

Usage:
    python draft/team_projections.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

SHRINKAGE_GAMES = 4  # pseudo-games' worth of weight pulled toward league average


def project_teams(season: int) -> list[dict]:
    prior_season = season - 1
    print(f"Projecting {season} team scoring from {prior_season} results...")

    sched = nfl.import_schedules([prior_season])
    played = sched.dropna(subset=["home_score", "away_score"])

    if played.empty:
        print(f"No completed games found for {prior_season} yet.")
        return []

    home = played[["home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points"})
    away = played[["away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points"})
    scores = pd.concat([home, away], ignore_index=True)

    team_stats = scores.groupby("team")["points"].agg(["sum", "count"]).reset_index()
    league_avg_ppg = scores["points"].mean()

    rows = []
    for _, r in team_stats.iterrows():
        # Bayesian-style shrinkage: blend the team's own total with
        # SHRINKAGE_GAMES worth of league-average games.
        blended_ppg = (r["sum"] + SHRINKAGE_GAMES * league_avg_ppg) / (r["count"] + SHRINKAGE_GAMES)
        rows.append({
            "season": season,
            "team": r["team"],
            "proj_ppg": round(blended_ppg, 2),
            "source": "shrinkage_prior_season",
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = project_teams(args.season)
    conn = get_connection()
    upsert_rows(conn, "team_projections", rows)
    conn.close()
