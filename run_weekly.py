"""
Runs the entire weekly routine in the correct order: pulls this week's
data, backfills last week's actual results as training labels, rebuilds
features, retrains the model, generates projections, computes team
needs, and prints the optimal lineup.

This is the single command to run once a week during the season. Each
step is logged with its own pass/fail status - a failure in one step
(e.g. the odds API being briefly down) doesn't stop the rest of the
pipeline from running, since most steps still produce useful output
with slightly degraded inputs. Steps that MUST succeed for anything
downstream to work (stats, features, prediction) are marked critical
and do stop the run if they fail.

--completed-week and --upcoming-week default to "auto": detected from
the schedule table as the most recent week where every game has a
final score, and the week after it. This is what makes unattended
scheduling (cron, GitHub Actions) possible without hardcoding a week
number that goes stale.

Usage:
    python run_weekly.py --season 2026 --fantasy-team-id 3
    python run_weekly.py --season 2026 --completed-week 5 --upcoming-week 6 --fantasy-team-id 3
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from db.connection import get_connection

ROOT = Path(__file__).parent
PYTHON = sys.executable


def detect_weeks(season: int) -> tuple[int, int]:
    """Most recent week where every scheduled game has a final score is
    'completed'; the week after that is 'upcoming'. Falls back to week
    1/1 if the schedule has no completed games yet (start of season)."""
    conn = get_connection()
    import pandas as pd
    sched = pd.read_sql(
        "SELECT week, home_score, away_score FROM schedule WHERE season = ?",
        conn, params=(season,),
    )
    conn.close()

    if sched.empty:
        raise RuntimeError(
            f"No schedule found for {season}. Run ingest/schedule.py first - "
            f"auto week detection needs it."
        )

    sched["complete"] = sched["home_score"].notna() & sched["away_score"].notna()
    fully_complete_weeks = sched.groupby("week")["complete"].all()
    completed = fully_complete_weeks[fully_complete_weeks].index.max()

    if pd.isna(completed):
        return 0, 1  # nothing played yet - preseason, project week 1
    completed = int(completed)
    upcoming = min(completed + 1, int(sched["week"].max()))
    return completed, upcoming


def run_step(description: str, args: list[str], critical: bool = False) -> bool:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    start = time.time()
    result = subprocess.run([PYTHON] + args, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"-- OK ({elapsed:.1f}s)")
        return True

    status = "FAILED (critical)" if critical else "FAILED (non-critical, continuing)"
    print(f"-- {status}")
    if critical:
        print(f"\nStopping: '{description}' is required for later steps to work.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--completed-week", type=int, default=None,
                         help="The week that just finished. Default: auto-detect")
    parser.add_argument("--upcoming-week", type=int, default=None,
                         help="The week to project and build a lineup for. Default: auto-detect")
    parser.add_argument("--fantasy-team-id", required=True)
    args = parser.parse_args()

    if args.completed_week is None or args.upcoming_week is None:
        print("Refreshing schedule before auto-detecting the current week...")
        if not run_step("Pull/refresh season schedule",
                         ["ingest/schedule.py", "--season", str(args.season)], critical=True):
            sys.exit(1)
        detected_cw, detected_uw = detect_weeks(args.season)
        completed_week = args.completed_week or detected_cw
        upcoming_week = args.upcoming_week or detected_uw
        print(f"Auto-detected weeks: completed={completed_week}, upcoming={upcoming_week}")
    else:
        completed_week, upcoming_week = args.completed_week, args.upcoming_week

    S, CW, UW, TEAM = args.season, completed_week, upcoming_week, args.fantasy_team_id
    preseason = (CW == 0)
    if preseason:
        print("\nNo completed weeks yet this season (preseason) - skipping stats "
              "pull and backfill, since there's nothing to fetch. Weekly "
              "projections won't be meaningful until a few weeks of real games "
              "exist; use predict_preseason.py for draft-time projections instead.")
    run_start = time.time()

    steps = []
    if not preseason:
        steps += [
            ("Pull completed week's stats",
             ["ingest/stats.py", "--season", str(S), "--week", str(CW)], True),
            ("Backfill completed week's results as training labels",
             ["features/backfill_labels.py", "--season", str(S), "--week", str(CW)], False),
        ]
    steps += [
        ("Pull upcoming week's injury reports",
         ["ingest/injuries.py", "--season", str(S), "--week", str(UW)], False),
        ("Pull upcoming week's odds",
         ["ingest/odds.py", "--season", str(S), "--week", str(UW)], False),
        ("Pull league settings and rosters",
         ["ingest/espn_league.py", "--season", str(S), "--week", str(UW)], True),
        ("Refresh ESPN/nflverse player ID mapping",
         ["features/id_mapping.py", "--season", str(S)], False),
        ("Build features for the upcoming week",
         ["features/build_features.py", "--season", str(S), "--week", str(UW)], True),
        ("Apply injury risk for the upcoming week",
         ["features/build_injury_risk.py", "--season", str(S), "--week", str(UW)], False),
        ("Retrain the weekly projection model",
         ["models/train_projection.py", "--season", str(S)], True),
        ("Generate projections for the upcoming week",
         ["models/predict_projection.py", "--season", str(S), "--week", str(UW)], True),
        ("Compute team needs",
         ["features/build_team_needs.py", "--season", str(S), "--week", str(UW),
          "--fantasy-team-id", TEAM], False),
        ("Solve the optimal lineup",
         ["models/lineup_optimizer.py", "--season", str(S), "--week", str(UW),
          "--fantasy-team-id", TEAM], False),
    ]

    results = []
    for description, script_args, critical in steps:
        ok = run_step(description, script_args, critical)
        results.append((description, ok, critical))
        if not ok and critical:
            break

    elapsed = time.time() - run_start
    print(f"\n{'='*60}\nSummary ({elapsed:.1f}s total)\n{'='*60}")
    for description, ok, critical in results:
        mark = "OK  " if ok else ("FAIL" if critical else "warn")
        print(f"  [{mark}] {description}")

    failed_critical = any(not ok for _, ok, critical in results if critical)
    if failed_critical:
        print("\nRun stopped early due to a critical failure - see above.")
        sys.exit(1)
    print("\nDone. Run models/recommend.py separately when you want draft/"
          "waiver/trade suggestions - that's a lookup, not a weekly batch step.")


if __name__ == "__main__":
    main()
