"""
One-time setup for draft recommendations: pulls several past seasons of
historical data, fits the rookie draft-capital curve, builds and labels
preseason features across those seasons, trains the preseason model,
and generates this year's draft projections.

Run this once, well before your draft. It does NOT need to be re-run
weekly - unlike run_weekly.py, this only depends on data that doesn't
change during the season (career history, past team environments).
Re-running it next year (or if you want to add another historical
season to improve the model) is fine and safe - everything is upsert
based.

This can take several minutes - it's pulling 18 weeks of stats for each
historical season.

Usage:
    python setup_draft_model.py --historical-seasons 2023 2024 2025 --target-season 2026
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = sys.executable


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
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-seasons", type=int, nargs="+", required=True,
                         help="Past COMPLETED seasons to build training data from, "
                              "e.g. 2023 2024 2025 (need at least 2)")
    parser.add_argument("--target-season", type=int, required=True,
                         help="The season you're drafting for, e.g. 2026")
    args = parser.parse_args()

    seasons = sorted(args.historical_seasons)
    target = args.target_season
    if len(seasons) < 2:
        print("Need at least 2 historical seasons to train a meaningful model.")
        sys.exit(1)

    run_start = time.time()
    results = []

    # --- Step 1: pull full historical data for each past season ---
    for season in seasons:
        ok = run_step(f"Pull schedule for {season}",
                       ["ingest/schedule.py", "--season", str(season)], True)
        results.append((f"Schedule {season}", ok, True))
        if not ok:
            sys.exit(1)

        for week in range(1, 19):
            ok = run_step(f"Pull stats: {season} week {week}",
                           ["ingest/stats.py", "--season", str(season), "--week", str(week)],
                           False)
            results.append((f"Stats {season} wk{week}", ok, False))

        ok = run_step(f"Build team season stats for {season}",
                       ["features/build_team_season_stats.py", "--season", str(season)], True)
        results.append((f"Team stats {season}", ok, True))

        ok = run_step(f"Pull draft picks for {season}",
                       ["ingest/draft_picks.py", "--season", str(season)], False)
        results.append((f"Draft picks {season}", ok, False))

    # Target season's own draft class and schedule
    run_step(f"Pull draft picks for {target}",
              ["ingest/draft_picks.py", "--season", str(target)], False)
    run_step(f"Pull schedule for {target}",
              ["ingest/schedule.py", "--season", str(target)], True)

    # --- Step 2: fit the rookie draft-capital curve ---
    run_step("Fit rookie baseline curve",
              ["features/build_rookie_baseline.py"], False)

    # --- Step 3: build + label preseason features for each season after the first ---
    # (each needs the season before it as "prior" data, so the earliest
    # historical season only serves as prior data - it doesn't get its own label)
    for i in range(1, len(seasons)):
        label_season = seasons[i]
        prior_season = seasons[i - 1]

        ok = run_step(f"Build schedule strength for {label_season} (based on {prior_season} defense)",
                       ["features/build_schedule_strength.py",
                        "--target-season", str(label_season),
                        "--stats-season", str(prior_season)], True)
        if not ok:
            continue

        run_step(f"Build preseason features for {label_season}",
                  ["features/build_preseason_features.py", "--season", str(label_season)], True)
        run_step(f"Backfill actual {label_season} results as labels",
                  ["features/backfill_preseason_labels.py", "--season", str(label_season)], True)

    # --- Step 4: build target season's own (unlabeled) features ---
    run_step(f"Build schedule strength for {target} (based on {seasons[-1]} defense)",
              ["features/build_schedule_strength.py",
               "--target-season", str(target), "--stats-season", str(seasons[-1])], True)
    run_step(f"Build preseason features for {target}",
              ["features/build_preseason_features.py", "--season", str(target)], True)

    # --- Step 5: train and predict ---
    train_ok = run_step("Train the preseason model",
                         ["models/train_preseason.py"], True)
    if train_ok:
        run_step(f"Generate {target} draft projections",
                  ["models/predict_preseason.py", "--season", str(target)], True)

    elapsed = time.time() - run_start
    print(f"\n{'='*60}\nDone in {elapsed/60:.1f} minutes\n{'='*60}")
    print("Run 'python models/predict_preseason.py --season "
          f"{target}' any time to re-print the ranked draft board.")


if __name__ == "__main__":
    main()
