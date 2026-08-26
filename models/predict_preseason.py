"""
Loads the trained preseason model and generates predicted season-long
average fantasy points per game for every player with preseason
features built for the target season, then converts those raw
projections into VALUE OVER REPLACEMENT (VOR).

Raw predicted_ppg is NOT what a draft board should be sorted by - QBs
score more raw points than any other position in most scoring formats,
but you can only start one, so the marginal QB is still quite good
while the marginal RB/WR is much worse than the elite ones. VOR
corrects for this: it's how much better a player is than the
replacement-level player at their position, given your league's actual
size and starter requirements. THIS is what draft order should follow.

Prints a PER-POSITION board (top N at each position) rather than one
mixed list - much closer to how an actual draft board gets used.

Usage:
    python models/predict_preseason.py --season 2026
    python models/predict_preseason.py --season 2026 --league-size 10 --per-position 8
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows
from models.preseason_feature_config import prepare_x
from features.build_team_needs import starter_slots_by_position

MODEL_PATH = Path(__file__).parent / "artifacts" / "preseason_model.json"


def get_num_teams(season: int, override: int | None) -> int:
    if override is not None:
        return override
    conn = get_connection()
    row = pd.read_sql(
        "SELECT num_teams FROM league_info WHERE season = ?", conn, params=(season,)
    )
    conn.close()
    if not row.empty and pd.notna(row.iloc[0]["num_teams"]):
        return int(row.iloc[0]["num_teams"])
    print("No league size found in league_info (run ingest/espn_league.py, "
          "or pass --league-size) - defaulting to 12.")
    return 12


def add_vor(df: pd.DataFrame, season: int, num_teams: int) -> pd.DataFrame:
    """VOR = predicted_ppg minus the replacement-level player's ppg at
    that position. Replacement level = the Nth best player at that
    position, where N = num_teams * starters_needed_at_that_position
    (FLEX slots split across RB/WR/TE, same logic used for team needs)."""
    slots = starter_slots_by_position(season)
    df = df.copy()
    df["vor"] = 0.0

    skipped_positions = []
    for position, starters_per_team in slots.items():
        if starters_per_team <= 0:
            skipped_positions.append(position)
            continue
        pos_df = df[df["position"] == position].sort_values(
            "predicted_ppg", ascending=False)
        if pos_df.empty:
            continue

        replacement_rank = max(1, round(num_teams * starters_per_team))
        if replacement_rank > len(pos_df):
            replacement_rank = len(pos_df)  # not enough players - use the worst available
        baseline = pos_df.iloc[replacement_rank - 1]["predicted_ppg"]

        df.loc[df["position"] == position, "vor"] = (
            df.loc[df["position"] == position, "predicted_ppg"] - baseline
        )

    if skipped_positions:
        print(f"\nWARNING: no starter slots configured for {skipped_positions} - "
              f"VOR will be 0 for these positions (they won't rank meaningfully). "
              f"This means league_roster_settings has no data for {season}. Run "
              f"'python ingest/espn_league.py --season {season} --week 1' to fix.\n")

    return df


def predict(season: int, league_size: int | None) -> list[dict]:
    if not MODEL_PATH.exists():
        print(f"No trained preseason model at {MODEL_PATH}. Run "
              f"train_preseason.py first.")
        return []

    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM preseason_features WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if df.empty:
        print(f"No preseason_features found for {season}. Run "
              f"build_preseason_features.py for this season first.")
        return []

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    X = prepare_x(df)
    df["predicted_ppg"] = model.predict(X).round(2)

    num_teams = get_num_teams(season, league_size)
    df = add_vor(df, season, num_teams)
    df["vor"] = df["vor"].round(2)

    cols = ["season", "player_id", "player_name", "position", "team",
            "predicted_ppg", "vor"]
    return df[cols].to_dict("records")


def print_board(rows: list[dict], per_position: int) -> None:
    by_position: dict[str, list[dict]] = {}
    for r in rows:
        by_position.setdefault(r["position"], []).append(r)

    # Print positions in a sensible draft order rather than alphabetical
    position_order = ["QB", "RB", "WR", "TE", "K", "D/ST"]
    ordered = [p for p in position_order if p in by_position]
    ordered += [p for p in by_position if p not in ordered]  # anything unexpected, at the end

    print(f"\n{'='*60}\nDraft board (top {per_position} per position, by VOR)\n{'='*60}")
    for position in ordered:
        players = sorted(by_position[position], key=lambda r: r["vor"], reverse=True)
        print(f"\n{position}:")
        for r in players[:per_position]:
            print(f"  {r['player_name']:<25} {r['predicted_ppg']:>6.2f} pts/gm   "
                  f"VOR: {r['vor']:>+6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--league-size", type=int, default=None,
                         help="Number of teams in your league. Overrides "
                              "league_info if provided.")
    parser.add_argument("--per-position", type=int, default=10,
                         help="How many players to show per position (default 10)")
    args = parser.parse_args()

    rows = predict(args.season, args.league_size)
    if rows:
        conn = get_connection()
        upsert_rows(conn, "preseason_projections", rows)
        conn.close()
        print_board(rows, args.per_position)
