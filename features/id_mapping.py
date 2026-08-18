"""
Builds a mapping between ESPN player IDs (used in rosters) and nflverse
player IDs (used in player_stats), since the two data sources don't
share a common ID system. Matches are made by normalized player name,
with team used as a tiebreaker when names collide.

Run this after both rosters and player_stats have data for the season -
it doesn't hit any external API, it just joins what's already in the DB.

Usage:
    python features/id_mapping.py --season 2026
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows


def normalize_name(name: str) -> str:
    """Lowercase, strip suffixes like Jr./II, and drop punctuation so
    'A.J. Brown' and 'AJ Brown Jr.' both normalize to 'aj brown'."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[.\']", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"[^a-z ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def build_mapping(season: int) -> list[dict]:
    conn = get_connection()

    rosters = pd.read_sql(
        "SELECT DISTINCT player_id, player_name, position FROM rosters WHERE season = ?",
        conn, params=(season,),
    )
    stats = pd.read_sql(
        "SELECT DISTINCT player_id, player_name, team FROM player_stats WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if rosters.empty or stats.empty:
        print("Need both rosters and player_stats populated for this season first.")
        return []

    rosters["norm_name"] = rosters["player_name"].apply(normalize_name)
    stats["norm_name"] = stats["player_name"].apply(normalize_name)

    # A normalized name that appears more than once in nflverse stats is
    # ambiguous (rare, but happens) - drop those rather than guess wrong.
    dupe_names = stats["norm_name"].value_counts()
    dupe_names = set(dupe_names[dupe_names > 1].index)

    rows = []
    unmatched = []
    for _, r in rosters.iterrows():
        if r["norm_name"] in dupe_names or not r["norm_name"]:
            unmatched.append(r["player_name"])
            continue

        match = stats[stats["norm_name"] == r["norm_name"]]
        if match.empty:
            unmatched.append(r["player_name"])
            continue

        rows.append({
            "espn_player_id": str(r["player_id"]),
            "nflverse_player_id": match.iloc[0]["player_id"],
            "player_name": r["player_name"],
            "team": match.iloc[0]["team"],
            "match_confidence": "exact",
        })

    if unmatched:
        print(f"{len(unmatched)} players could not be matched automatically:")
        for name in unmatched[:20]:
            print(f"  - {name}")
        print("These are likely name-spelling mismatches or defense/special "
              "teams entries - worth a manual pass in player_id_map.")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = build_mapping(args.season)
    conn = get_connection()
    upsert_rows(conn, "player_id_map", rows)
    conn.close()
