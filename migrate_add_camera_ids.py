"""
Migration: add `camera_ids` TEXT column to the jobs table.

This column stores a JSON array of per-camera entries so a single batch
job can search footage from multiple cameras simultaneously:

  [
    {"camera_id": "...", "camera_name": "...", "start_time": "HH:MM:SS", "end_time": "HH:MM:SS"},
    ...
  ]

Run once with:
    python migrate_add_camera_ids.py
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

    if 'camera_ids' in columns:
        print("Column 'camera_ids' already exists in jobs. No migration needed.")
        conn.close()
        return

    print("Adding 'camera_ids' column to jobs table...")
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN camera_ids TEXT")
        conn.commit()
        print("Migration complete. 'camera_ids' column added to jobs.")
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