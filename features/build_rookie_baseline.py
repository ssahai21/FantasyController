"""
Fits a simple curve per position: given a player's overall draft pick,
what fantasy points per game should we expect in their ROOKIE season?
Fit on log(overall_pick) since draft capital value drops off steeply
at the top and flattens out later - pick 1 vs pick 10 is a much bigger
gap than pick 200 vs pick 210.

Needs draft_capital + player_stats for multiple historical draft
classes to fit a meaningful curve - the more rookie seasons of data,
the better calibrated this gets.

Usage:
    python features/build_rookie_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

MIN_SAMPLES_PER_POSITION = 8


def fit_curves() -> list[dict]:
    conn = get_connection()
    draft_capital = pd.read_sql(
        "SELECT draft_year, player_id, position, overall_pick FROM draft_capital "
        "WHERE overall_pick IS NOT NULL", conn,
    )
    conn.close()

    if draft_capital.empty:
        print("No draft_capital found. Run ingest/draft_picks.py for some "
              "historical draft years first.")
        return []

    rows = []
    for position, group in draft_capital.groupby("position"):
        # For each drafted player, get their actual rookie-season ppg -
        # i.e. their stats in the SAME season as their draft_year
        conn = get_connection()
        actuals = []
        for _, pick in group.iterrows():
            stats = pd.read_sql(
                "SELECT AVG(fantasy_points) as ppg FROM player_stats "
                "WHERE season = ? AND player_id = ?",
                conn, params=(pick["draft_year"], pick["player_id"]),
            )
            if not stats.empty and pd.notnull(stats.iloc[0]["ppg"]):
                actuals.append({
                    "overall_pick": pick["overall_pick"],
                    "ppg": stats.iloc[0]["ppg"],
                })
        conn.close()

        if len(actuals) < MIN_SAMPLES_PER_POSITION:
            print(f"Only {len(actuals)} {position} rookies with stats found - "
                  f"need at least {MIN_SAMPLES_PER_POSITION} to fit a curve. Skipping.")
            continue

        df = pd.DataFrame(actuals)
        log_pick = np.log(df["overall_pick"])
        slope, intercept = np.polyfit(log_pick, df["ppg"], 1)

        rows.append({
            "position": position,
            "intercept": round(float(intercept), 3),
            "slope": round(float(slope), 3),
            "sample_size": len(actuals),
        })
        print(f"{position}: ppg = {intercept:.2f} + {slope:.2f} * ln(pick)  (n={len(actuals)})")

    return rows


if __name__ == "__main__":
    rows = fit_curves()
    conn = get_connection()
    upsert_rows(conn, "rookie_baseline_curve", rows)
    conn.close()
