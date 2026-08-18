"""
Pulls current NFL spreads and totals from The Odds API and writes
them into game_odds, including derived implied team totals.

Requires an API key from https://the-odds-api.com (free tier: 500
requests/month) set as ODDS_API_KEY in a .env file.

Usage:
    python ingest/odds.py --season 2026 --week 5
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
API_KEY = os.getenv("ODDS_API_KEY")
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def fetch_current_odds(season: int, week: int) -> list[dict]:
    if not API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY not set. Add it to a .env file in the project root."
        )

    print(f"Fetching odds for {season} week {week}...")
    resp = requests.get(ODDS_URL, params={
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    })
    resp.raise_for_status()
    games = resp.json()

    rows = []
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]

        spread = None
        total = None
        # Average the line across available bookmakers for stability
        spreads, totals = [], []
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_team:
                            spreads.append(outcome["point"])
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        totals.append(outcome["point"])

        if spreads:
            spread = round(sum(spreads) / len(spreads), 1)
        if totals:
            total = round(sum(totals) / len(totals), 1)

        home_implied = away_implied = None
        if spread is not None and total is not None:
            home_implied = round((total / 2) - (spread / 2), 1)
            away_implied = round((total / 2) + (spread / 2), 1)

        rows.append({
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "spread": spread,
            "total": total,
            "home_implied_total": home_implied,
            "away_implied_total": away_implied,
        })

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    rows = fetch_current_odds(args.season, args.week)
    conn = get_connection()
    upsert_rows(conn, "game_odds", rows)
    conn.close()