"""
Shared SQLite connection helper.

Every ingestion script imports get_connection() from here so they all
write to the same database file and the schema only has to be applied once.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "fantasy.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Columns added to existing tables AFTER that table was first created.
# CREATE TABLE IF NOT EXISTS (in schema.sql) only helps for brand new
# tables - it does nothing for a table that already exists but is
# missing a column added later. Rather than writing a one-off
# migrate_*.py script every time this happens (it's happened three
# times now), new columns go here instead and self-apply automatically
# on every connection. Format: (table, column, "TYPE [DEFAULT ...]").
COLUMN_MIGRATIONS = [
    ("schedule", "home_score", "INTEGER"),
    ("schedule", "away_score", "INTEGER"),
    ("preseason_projections", "vor", "REAL"),
    ("preseason_features", "age", "REAL"),
    ("preseason_features", "is_rookie", "INTEGER"),
]


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for table, column, coltype in COLUMN_MIGRATIONS:
        existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table doesn't exist yet - schema.sql will create it
                      # fresh with all current columns, nothing to migrate
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            print(f"Migrated: added {table}.{column}")
    conn.commit()


def get_connection() -> sqlite3.Connection:
    """Open a connection to the pipeline's SQLite database, creating
    the data directory and database file if needed, applying the full
    schema (safe/idempotent - CREATE TABLE IF NOT EXISTS), then applying
    any column-level migrations for tables that existed before a column
    was added to them."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)

    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    if is_new:
        print(f"Initialized new database at {DB_PATH}")

    _apply_column_migrations(conn)

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
