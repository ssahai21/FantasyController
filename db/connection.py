"""
Shared SQLite connection helper.

Every ingestion script imports get_connection() from here so they all
write to the same database file and the schema only has to be applied once.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "fantasy.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Open a connection to the pipeline's SQLite database, creating
    the data directory, the database file, and applying the schema on
    first use. The directory creation matters specifically for fresh
    checkouts (e.g. GitHub Actions) - git doesn't track empty folders,
    so data/ may not exist at all until this creates it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)

    if is_new:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
        print(f"Initialized new database at {DB_PATH}")

    return conn


def upsert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    """Insert rows into `table`, replacing any existing row with the
    same primary key. `rows` is a list of dicts with matching column names."""
    if not rows:
        print(f"No rows to write to {table}")
        return

    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    col_list = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"

    values = [tuple(row[c] for c in columns) for row in rows]
    conn.executemany(sql, values)
    conn.commit()
    print(f"Wrote {len(rows)} rows to {table}")
