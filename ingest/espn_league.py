"""
Pulls league scoring settings, roster slot requirements, and team
rosters from a private ESPN fantasy football league, using the
community-maintained espn_api package rather than hand-rolled requests
against ESPN's undocumented API. ESPN's anti-bot protection (Akamai)
blocks plain requests-library traffic even with valid cookies - espn_api
is a widely used, actively maintained library that already handles this
correctly, so it's a much better bet than re-implementing auth by hand.

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

from dotenv import load_dotenv
from espn_api.football import League

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

load_dotenv()
ESPN_S2 = os.getenv("ESPN_S2")
ESPN_SWID = os.getenv("ESPN_SWID")
LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID")

# espn_api's roster_slots dict uses its own position labels, which
# don't always match nflverse's position codes used elsewhere in the
# pipeline (e.g. "RB/WR" for a shared flex-style slot, "D/ST" naming).
# This maps the common ones - anything not in this map falls through
# unchanged rather than being silently dropped.
SLOT_NAME_MAP = {
    "RB/WR": "FLEX", "RB/WR/TE": "FLEX", "WR/TE": "FLEX",
    "D/ST": "D/ST", "DST": "D/ST",
}

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "D/ST"}


def get_league(season: int) -> League:
    if not (ESPN_S2 and ESPN_SWID and LEAGUE_ID):
        raise RuntimeError(
            "ESPN_S2, ESPN_SWID, and ESPN_LEAGUE_ID must all be set in .env"
        )
    return League(league_id=int(LEAGUE_ID), year=season,
                  espn_s2=ESPN_S2, swid=ESPN_SWID)


def fetch_league_settings(league: League, season: int) -> list[dict]:
    print(f"Fetching league scoring settings for {season}...")
    scoring = getattr(league.settings, "scoring_format", [])

    # scoring_format entries look like {'points': X, 'label': '...',
    # 'abbr': '...'} - the label is the closest thing to a stable
    # category name espn_api exposes, so we key off that rather than
    # ESPN's internal numeric stat IDs.
    rows = []
    for item in scoring:
        label = item.get("label") or item.get("abbr")
        points = item.get("points")
        if label is None or points is None:
            continue
        rows.append({
            "season": season,
            "stat_category": label,
            "points_per_unit": points,
        })
    return rows


def fetch_roster_settings(league: League, season: int) -> list[dict]:
    print(f"Fetching roster slot requirements for {season}...")
    slots = getattr(league.settings, "roster_slots", {}) or {}

    rows = []
    for slot_name, count in slots.items():
        if not count or count <= 0:
            continue
        mapped_name = SLOT_NAME_MAP.get(slot_name, slot_name)
        rows.append({"season": season, "lineup_slot": mapped_name, "slot_count": count})
    return rows


def fetch_rosters(league: League, season: int, week: int) -> list[dict]:
    print(f"Fetching rosters for {season} week {week}...")
    rows = []
    for team in league.teams:
        for player in team.roster:
            position = getattr(player, "position", None)
            slot = getattr(player, "lineupSlot", None)
            rows.append({
                "season": season,
                "week": week,
                "fantasy_team_id": str(team.team_id),
                "fantasy_team_name": team.team_name,
                "player_id": str(player.playerId),
                "player_name": player.name,
                "position": position,
                "lineup_slot": SLOT_NAME_MAP.get(slot, slot) if slot else position,
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    league = get_league(args.season)

    conn = get_connection()
    upsert_rows(conn, "league_settings", fetch_league_settings(league, args.season))
    upsert_rows(conn, "league_roster_settings", fetch_roster_settings(league, args.season))
    upsert_rows(conn, "rosters", fetch_rosters(league, args.season, args.week))
    conn.close()
