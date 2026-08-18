"""
Pulls league scoring settings and team rosters from a private ESPN
fantasy football league, using the unofficial ESPN fantasy API.

Requires espn_s2 and SWID cookies (see README for how to find them)
plus your league ID, set in .env as:
    ESPN_S2=...
    ESPN_SWID={...}
    ESPN_LEAGUE_ID=123456

Usage:
    python ingest/espn_league.py --season 2026 --week 5
"""
import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

load_dotenv()
ESPN_S2 = os.getenv("ESPN_S2")
ESPN_SWID = os.getenv("ESPN_SWID")
LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID")

BASE_URL = "https://fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}"

# ESPN's scoring stat IDs mapped to the category names we use elsewhere
# in the pipeline. This is not the full list - ESPN has ~90 stat IDs -
# just the ones that map to standard fantasy scoring categories.
ESPN_STAT_ID_MAP = {
    "3": "passing_yards", "4": "pass_tds", "20": "interceptions",
    "24": "rush_yards", "25": "rush_tds",
    "42": "rec_yards", "43": "rec_tds", "53": "receptions",
    "72": "fumbles_lost",
}

LINEUP_SLOT_MAP = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST",
    17: "K", 20: "BENCH", 21: "IR", 23: "FLEX",
}


def _session():
    if not (ESPN_S2 and ESPN_SWID and LEAGUE_ID):
        raise RuntimeError(
            "ESPN_S2, ESPN_SWID, and ESPN_LEAGUE_ID must all be set in .env"
        )
    s = requests.Session()
    s.cookies.update({"espn_s2": ESPN_S2, "SWID": ESPN_SWID})
    return s


def fetch_league_settings(season: int) -> list[dict]:
    print(f"Fetching league scoring settings for {season}...")
    url = BASE_URL.format(season=season, league_id=LEAGUE_ID)
    resp = _session().get(url, params={"view": "mSettings"})
    resp.raise_for_status()
    data = resp.json()

    scoring_items = (
        data.get("settings", {})
        .get("scoringSettings", {})
        .get("scoringItems", [])
    )

    rows = []
    for item in scoring_items:
        stat_id = str(item.get("statId"))
        category = ESPN_STAT_ID_MAP.get(stat_id)
        if category is None:
            continue  # skip stat IDs we haven't mapped (IDP, misc bonuses, etc.)
        rows.append({
            "season": season,
            "stat_category": category,
            "points_per_unit": item.get("points"),
        })

    return rows


def fetch_rosters(season: int, week: int) -> list[dict]:
    print(f"Fetching rosters for {season} week {week}...")
    url = BASE_URL.format(season=season, league_id=LEAGUE_ID)
    resp = _session().get(url, params={"view": "mRoster", "view": "mTeam"})
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for team in data.get("teams", []):
        team_id = str(team.get("id"))
        team_name = f"{team.get('location', '')} {team.get('nickname', '')}".strip()

        for entry in team.get("roster", {}).get("entries", []):
            player_info = entry.get("playerPoolEntry", {}).get("player", {})
            rows.append({
                "season": season,
                "week": week,
                "fantasy_team_id": team_id,
                "fantasy_team_name": team_name or team_id,
                "player_id": str(player_info.get("id")),
                "player_name": player_info.get("fullName"),
                "position": player_info.get("defaultPositionId"),
                "lineup_slot": LINEUP_SLOT_MAP.get(entry.get("lineupSlotId"), "UNKNOWN"),
            })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    conn = get_connection()
    upsert_rows(conn, "league_settings", fetch_league_settings(args.season))
    upsert_rows(conn, "rosters", fetch_rosters(args.season, args.week))
    conn.close()
