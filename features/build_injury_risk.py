"""
Applies the calibrated play probabilities from calibrate_injury_risk.py
to this week's actual injury reports, producing one play probability
per player. Players with no injury report at all default to 0.98 -
not 1.0, since anyone can get hurt in warmups, but treated as a
fundamentally different (much safer) population than anyone who
actually appears on the report.

Usage:
    python features/build_injury_risk.py --season 2026 --week 6
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

HEALTHY_DEFAULT = 0.98
FALLBACK_RATE = 0.75  # used if a status appears with no calibrated rate yet


def build(season: int, week: int) -> list[dict]:
    conn = get_connection()
    injuries = pd.read_sql(
        "SELECT player_id, report_status FROM injury_reports WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    rates = pd.read_sql("SELECT report_status, play_probability FROM injury_risk_rates", conn)
    roster_players = pd.read_sql(
        "SELECT DISTINCT player_id FROM features WHERE season = ? AND week = ?",
        conn, params=(season, week),
    )
    conn.close()

    if roster_players.empty:
        print(f"No features found for {season} week {week}. Run "
              f"build_features.py first so there's a player list to apply risk to.")
        return []

    rate_map = dict(zip(rates["report_status"], rates["play_probability"]))

    injured = injuries.merge(
        pd.Series(rate_map, name="play_probability").rename_axis("report_status").reset_index(),
        on="report_status", how="left",
    )
    injured["play_probability"] = injured["play_probability"].fillna(FALLBACK_RATE)

    all_players = roster_players.merge(injured, on="player_id", how="left")
    all_players["play_probability"] = all_players["play_probability"].fillna(HEALTHY_DEFAULT)
    all_players["report_status"] = all_players["report_status"].fillna("Healthy")
    all_players["season"] = season
    all_players["week"] = week

    return all_players[["season", "week", "player_id", "report_status",
                         "play_probability"]].to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = build(args.season, args.week)
    conn = get_connection()
    upsert_rows(conn, "injury_risk", rows)
    conn.close()
    print(f"Built injury risk for {len(rows)} players.")
