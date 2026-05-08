"""
Migration: add `selected_track_id` TEXT column to the jobs table.

Stores the track_id most recently used when generating a court report for
a given job, so the system can remember the officer's choice across sessions.
The column is nullable — existing rows and jobs with no allocated track store NULL.

Run once with:
    python migrate_add_selected_track_id.py
"""
import sqlite3
import os

DB_PATH = os.path.join('instance', 'system.sqlite')


def migrate(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(jobs)")
    columns = [row[1] for row in cur.fetchall()]

    if 'selected_track_id' in columns:
        print("Column 'selected_track_id' already exists in jobs. No migration needed.")
        conn.close()
        return

    print("Adding 'selected_track_id' column to jobs table...")
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN selected_track_id TEXT")
        conn.commit()
        print("Migration complete. 'selected_track_id' column added to jobs.")
    except Exception as e:
        conn.rollback()
        print(f"Migration FAILED: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
    else:
        migrate(DB_PATH)
