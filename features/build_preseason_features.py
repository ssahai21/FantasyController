"""
Builds draft-time features for a target season: each player's recent
career performance, their team's offensive environment (proxied from
the prior season), their full-season schedule strength, and their age
- combined into one row per player, usable before a single game of the
target season is played.

Only includes players who are actually on an NFL roster for the target
season (via nflreadpy's roster data) - this is what keeps retired or
long-gone players out of the draft board, since merely having played
in a past season isn't the same as being currently draftable.

Run build_team_season_stats.py and build_schedule_strength.py for the
relevant prior season(s) before this.

Usage:
    python features/build_preseason_features.py --season 2026
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import nflreadpy as nfl

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection, upsert_rows

# How many prior seasons of player performance to consider, and how to
# weight them - most recent season counts more than the one before it.
SEASON_WEIGHTS = {0: 0.65, 1: 0.35}  # 0 = most recent prior season


def player_season_averages(season: int) -> pd.DataFrame:
    """One row per player for a given completed season: games played
    and average fantasy points per game that season."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT player_id, player_name, position, team, week, fantasy_points "
        "FROM player_stats WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "position",
                                       "team", "games_played", "ppg"])

    agg = df.groupby("player_id").agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        games_played=("week", "nunique"),
        ppg=("fantasy_points", "mean"),
    ).reset_index()
    return agg


def fetch_active_roster_and_age(season: int) -> pd.DataFrame:
    """Pulls the target season's actual NFL rosters. Being present here
    at all is what filters out retired/no-longer-active players - if
    someone isn't on this list, they're not on an NFL roster for this
    season, regardless of what they did in a prior one. Also computes
    age as of the season, which is a real predictive signal (RBs decline
    early, WRs and QBs later) that wasn't being used before.

    NOTE: nflreadpy's exact column names have shifted between versions
    in this project's history (see stats.py/espn_league.py migration
    notes) - this checks a couple of likely names defensively rather
    than assuming one. If both checks fail, age comes back all-NaN
    (harmless - XGBoost handles missing values natively) rather than
    crashing the whole build.
    """
    try:
        rosters = nfl.load_rosters([season]).to_pandas()
    except Exception as e:
        print(f"Could not load {season} rosters ({e}) - skipping active-roster "
              f"filter and age for this build. Retired players may leak through.")
        return pd.DataFrame(columns=["player_id", "age"])

    id_col = "gsis_id" if "gsis_id" in rosters.columns else "player_id"
    if id_col not in rosters.columns:
        print(f"Roster data has no gsis_id/player_id column (columns: "
              f"{list(rosters.columns)}) - skipping active-roster filter.")
        return pd.DataFrame(columns=["player_id", "age"])

    rosters = rosters.rename(columns={id_col: "player_id"})

    if "age" in rosters.columns:
        rosters["age"] = pd.to_numeric(rosters["age"], errors="coerce")
    elif "birth_date" in rosters.columns:
        birth = pd.to_datetime(rosters["birth_date"], errors="coerce")
        season_start = pd.Timestamp(year=season, month=9, day=1)
        rosters["age"] = ((season_start - birth).dt.days / 365.25).round(1)
    else:
        print("No age or birth_date column in roster data - age will be "
              "missing for this build (active-roster filter still applies).")
        rosters["age"] = None

    return rosters[["player_id", "age"]].drop_duplicates(subset="player_id")


def rookie_rows(season: int, existing_player_ids: set, active_roster: pd.DataFrame) -> pd.DataFrame:
    """Rookies have no prior-season player_stats, so they're missing
    entirely from the normal preseason_features build. This adds them
    back in using the draft capital curve as their baseline instead of
    career history."""
    conn = get_connection()
    rookies = pd.read_sql(
        "SELECT player_id, player_name, position, team, overall_pick "
        "FROM draft_capital WHERE draft_year = ? AND overall_pick IS NOT NULL",
        conn, params=(season,),
    )
    curves = pd.read_sql("SELECT * FROM rookie_baseline_curve", conn)
    team_stats = pd.read_sql(
        "SELECT team, points_per_game FROM team_season_stats WHERE season = ?",
        conn, params=(season - 1,),
    )
    sched_strength = pd.read_sql(
        "SELECT team, position, avg_opp_def_rank FROM schedule_strength WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    rookies = rookies[~rookies["player_id"].isin(existing_player_ids)]
    if rookies.empty or curves.empty:
        return pd.DataFrame()

    rookies = rookies.merge(curves, on="position", how="inner")  # drop positions with no fitted curve
    rookies["two_season_avg_ppg"] = rookies["intercept"] + rookies["slope"] * np.log(rookies["overall_pick"])
    rookies["two_season_avg_ppg"] = rookies["two_season_avg_ppg"].clip(lower=0)  # no negative projections

    # ROOKIE DISCOUNT: the curve above is fit on a small, noisy sample
    # of past rookie classes, and a handful of standout rookie seasons
    # can pull the whole curve up. Real rookie transitions are usually
    # weaker than a raw draft-capital curve implies - especially at QB
    # and WR - so this shrinks the projection toward a more conservative
    # estimate rather than trusting the curve at full strength. 0.80 is
    # a deliberate, adjustable judgment call, not a derived number.
    ROOKIE_DISCOUNT = 0.80
    rookies["two_season_avg_ppg"] = rookies["two_season_avg_ppg"] * ROOKIE_DISCOUNT

    rookies = rookies.merge(
        team_stats.rename(columns={"points_per_game": "team_proj_ppg"}), on="team", how="left")
    rookies = rookies.merge(
        sched_strength.rename(columns={"avg_opp_def_rank": "schedule_strength_rank"}),
        on=["team", "position"], how="left")
    rookies = rookies.merge(active_roster, on="player_id", how="left")
    rookies["age"] = rookies["age"].fillna(22)  # typical rookie age fallback if roster age lookup missed them

    rookies["season"] = season
    rookies["is_rookie"] = 1
    rookies["last_season_ppg"] = None       # no prior NFL season exists
    rookies["games_played_last_season"] = 0
    rookies["fantasy_points_per_game"] = None

    cols = ["season", "player_id", "player_name", "position", "team", "age", "is_rookie",
            "last_season_ppg", "two_season_avg_ppg", "games_played_last_season",
            "team_proj_ppg", "schedule_strength_rank", "fantasy_points_per_game"]
    return rookies[cols].round(2)


def build(season: int) -> list[dict]:
    prior_seasons = [season - 1, season - 2]
    season_data = {i: player_season_averages(s) for i, s in enumerate(prior_seasons)}

    if season_data[0].empty:
        print(f"No player_stats found for {prior_seasons[0]} (most recent prior "
              f"season). Need at least one completed prior season to build "
              f"preseason features.")
        return []

    active_roster = fetch_active_roster_and_age(season)

    # Start from the most recent prior season's player list - that's
    # who's realistically draftable
    base = season_data[0][["player_id", "player_name", "position", "team",
                             "games_played", "ppg"]].rename(
        columns={"games_played": "games_played_last_season", "ppg": "last_season_ppg"})

    if not active_roster.empty:
        before = len(base)
        base = base.merge(active_roster[["player_id"]], on="player_id", how="inner")
        print(f"Active-roster filter: {before} -> {len(base)} players "
              f"(dropped players not on a {season} NFL roster)")

    # Weighted multi-season average, falling back to just last season's
    # number for players without two full prior seasons of data
    weighted = season_data[0][["player_id", "ppg"]].rename(columns={"ppg": "ppg_0"})
    if not season_data[1].empty:
        weighted = weighted.merge(
            season_data[1][["player_id", "ppg"]].rename(columns={"ppg": "ppg_1"}),
            on="player_id", how="left",
        )
    else:
        weighted["ppg_1"] = None

    weighted["two_season_avg_ppg"] = weighted.apply(
        lambda r: (
            r["ppg_0"] * SEASON_WEIGHTS[0] + r["ppg_1"] * SEASON_WEIGHTS[1]
            if pd.notnull(r["ppg_1"])
            else r["ppg_0"]  # no second season available, just use the one we have
        ),
        axis=1,
    )

    df = base.merge(weighted[["player_id", "two_season_avg_ppg"]], on="player_id", how="left")
    df = df.merge(active_roster, on="player_id", how="left")  # attaches age

    # Team environment proxy, from the most recent prior season
    conn = get_connection()
    team_stats = pd.read_sql(
        "SELECT team, points_per_game FROM team_season_stats WHERE season = ?",
        conn, params=(prior_seasons[0],),
    )
    sched_strength = pd.read_sql(
        "SELECT team, position, avg_opp_def_rank FROM schedule_strength WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    df = df.merge(
        team_stats.rename(columns={"points_per_game": "team_proj_ppg"}),
        on="team", how="left",
    )
    df = df.merge(sched_strength.rename(columns={"avg_opp_def_rank": "schedule_strength_rank"}),
                   on=["team", "position"], how="left")

    df["season"] = season
    df["is_rookie"] = 0
    df["fantasy_points_per_game"] = None  # filled in after the season, for training

    cols = ["season", "player_id", "player_name", "position", "team", "age", "is_rookie",
            "last_season_ppg", "two_season_avg_ppg", "games_played_last_season",
            "team_proj_ppg", "schedule_strength_rank", "fantasy_points_per_game"]
    df = df[cols].round(2).where(pd.notnull(df), None)

    rookies = rookie_rows(season, set(df["player_id"]), active_roster)
    if not rookies.empty:
        print(f"Added {len(rookies)} rookies via draft capital baseline.")
        rookies = rookies.where(pd.notnull(rookies), None)
        df = pd.concat([df, rookies], ignore_index=True)

    return df.to_dict("records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    rows = build(args.season)
    conn = get_connection()
    upsert_rows(conn, "preseason_features", rows)
    conn.close()
