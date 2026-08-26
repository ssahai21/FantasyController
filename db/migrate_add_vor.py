"""
One-time fix for anyone who ran predict_preseason.py before the vor
(value over replacement) column existed in the schema. Safe to re-run -
it checks first.

Usage:
    python db/migrate_add_vor.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from db.connection import get_connection

if __name__ == "__main__":
    conn = get_connection()
    cur = conn.cursor()
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(preseason_projections)")}

    if "vor" not in existing_cols:
        cur.execute("ALTER TABLE preseason_projections ADD COLUMN vor REAL")
        print("Added column vor to preseason_projections")
    else:
        print("vor already exists, skipping")

    conn.commit()
    conn.close()
