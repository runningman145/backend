"""
One-time migration: add 'cancelled' to the jobs status CHECK constraint.
SQLite does not support ALTER COLUMN, so we recreate the table.

Run once with:
    python migrate_add_cancelled_status.py
"""
import sqlite3
import os

# Adjust this path if your DB lives elsewhere
DB_PATH = os.path.join('instance', 'system.sqlite')

def migrate(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Confirm the constraint actually needs updating
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'")
    row = cur.fetchone()
    if row is None:
        print("No 'jobs' table found. Nothing to migrate.")
        conn.close()
        return

    if "'cancelled'" in row['sql']:
        print("Constraint already includes 'cancelled'. No migration needed.")
        conn.close()
        return

    print("Starting migration...")

    try:
        conn.execute("BEGIN")

        # 1. Rename existing table
        conn.execute("ALTER TABLE jobs RENAME TO jobs_old")

        # 2. Create new table with updated CHECK constraint
        conn.execute("""
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
                camera_id TEXT NOT NULL,
                detection_id INTEGER,
                video_filename TEXT,
                query_image_filename TEXT,
                threshold REAL DEFAULT 40,
                frame_skip INTEGER DEFAULT 15,
                job_date DATE,
                start_time TIME,
                end_time TIME,
                result_data TEXT,
                error_message TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        """)

        # 3. Copy all existing rows across
        conn.execute("""
            INSERT INTO jobs
            SELECT id, status, camera_id, detection_id, video_filename,
                   query_image_filename, threshold, frame_skip,
                   job_date, start_time, end_time,
                   result_data, error_message,
                   created_at, started_at, completed_at
            FROM jobs_old
        """)

        # 4. Drop the old table
        conn.execute("DROP TABLE jobs_old")

        conn.execute("COMMIT")
        print("Migration complete. 'cancelled' status is now allowed.")

    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"Migration FAILED and was rolled back: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    migrate(DB_PATH)