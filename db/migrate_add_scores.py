"""
One-time fix for anyone who ran ingest/schedule.py before home_score/
away_score existed in the schema. Safe to re-run - it checks first.

Usage:
    python db/migrate_add_scores.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection

if __name__ == "__main__":
    conn = get_connection()
    cur = conn.cursor()
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(schedule)")}

    for col in ("home_score", "away_score"):
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE schedule ADD COLUMN {col} INTEGER")
            print(f"Added column {col} to schedule")
        else:
            print(f"{col} already exists, skipping")

    conn.commit()
    conn.close()
