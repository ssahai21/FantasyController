"""
Builds the features table for a given week: rolling averages, usage
trend, opponent defensive strength, implied team total, and injury
status, joined together into one row per player.

Only uses data from weeks strictly before `week` for the rolling/trend/
opponent-strength calculations - the target week's own stats obviously
aren't known yet when you're building features to predict it.

Usage:
    python features/build_features.py --season 2026 --week 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

ROLLING_WINDOW = 3


def trend_slope(values: pd.Series) -> float:
    """Slope of a simple linear fit over the trailing weeks. Positive =
    usage trending up, negative = trending down. Returns 0 if too few
    data points to fit a meaningful trend."""
    values = values.dropna().tail(ROLLING_WINDOW)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values.values, 1)
    return round(float(slope), 3)


def build_rolling_stats(history: pd.DataFrame) -> pd.DataFrame:
    """One row per player: avg_pts_l3, avg_pts_season, target_share_trend,
    snap_pct_l3, computed from that player's games prior to the target week."""
    history = history.sort_values(["player_id", "week"])
    grouped = history.groupby("player_id")

    out = grouped.agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        avg_pts_season=("fantasy_points", "mean"),
    ).reset_index()

    l3 = grouped.apply(
        lambda g: pd.Series({
            "avg_pts_l3": g["fantasy_points"].tail(ROLLING_WINDOW).mean(),
            "snap_pct_l3": g["snap_pct"].tail(ROLLING_WINDOW).mean(),
            "target_share_trend": trend_slope(g["targets"]),
        }),
        include_groups=False,
    ).reset_index()

    return out.merge(l3, on="player_id", how="left")


def build_opponent_defense_rank(history: pd.DataFrame) -> pd.DataFrame:
    """Ranks each team 1 (toughest matchup, fewest points allowed) to
    32 (easiest) per position, based on fantasy points allowed to that
    position so far this season."""
    allowed = (
        history.groupby(["opponent", "position"])["fantasy_points"]
        .mean()
        .reset_index()
        .rename(columns={"opponent": "team", "fantasy_points": "pts_allowed"})
    )
    allowed["opp_def_rank_allowed"] = (
        allowed.groupby("position")["pts_allowed"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return allowed[["team", "position", "opp_def_rank_allowed"]]


def build_features(season: int, week: int) -> list[dict]:
    conn = get_connection()

    history = pd.read_sql(
        "SELECT * FROM player_stats WHERE season = ? AND week < ?",
        conn, params=(season, week),
    )
    if history.empty:
        print(f"No prior-week history found for {season} before week {week}.")
        conn.close()
        return []

    schedule = pd.read_sql(
        "SELECT * FROM schedule WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    odds = pd.read_sql(
        "SELECT * FROM game_odds WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    injuries = pd.read_sql(
        "SELECT player_id, report_status FROM injury_reports WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    conn.close()

    rolling = build_rolling_stats(history)
    def_rank = build_opponent_defense_rank(history)

    # Figure out each team's opponent for the target week from the schedule
    home = schedule[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"})
    away = schedule[["home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
    matchups = pd.concat([home, away], ignore_index=True)

    # Implied team total per team for the target week
    home_totals = odds[["home_team", "home_implied_total"]].rename(
        columns={"home_team": "team", "home_implied_total": "implied_team_total"})
    away_totals = odds[["away_team", "away_implied_total"]].rename(
        columns={"away_team": "team", "away_implied_total": "implied_team_total"})
    implied = pd.concat([home_totals, away_totals], ignore_index=True)

    df = rolling.merge(matchups, on="team", how="left")
    df = df.merge(implied, on="team", how="left")

    # def_rank is keyed on "team" meaning the defense - we want the rank
    # of the player's OPPONENT's defense, so merge it keyed on opponent
    df = df.merge(
        def_rank.rename(columns={"team": "opponent"}),
        on=["opponent", "position"], how="left",
    )
    df = df.merge(injuries, on="player_id", how="left")
    df = df.rename(columns={"report_status": "injury_status"})

    df["season"] = season
    df["week"] = week
    df["fantasy_points"] = None  # filled in after the week is played

    columns = [
        "season", "week", "player_id", "player_name", "position", "team",
        "opponent", "avg_pts_l3", "avg_pts_season", "target_share_trend",
        "snap_pct_l3", "opp_def_rank_allowed", "implied_team_total",
        "injury_status", "fantasy_points",
    ]
    df = df[columns].where(pd.notnull(df), None)
    return df.to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = build_features(args.season, args.week)
    conn = get_connection()
    upsert_rows(conn, "features", rows)
    conn.close()
