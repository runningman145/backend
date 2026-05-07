"""
Migration: Add query_embedding and job_id fields for proper cross-camera tracking.

This migration adds:
1. query_embedding BLOB field to vehicle_detections (stores the original query image embedding)
2. job_id TEXT field to vehicle_detections (links detection to its search job)
3. job_id TEXT field to vehicle_tracks (tracks originated from a specific search)

These changes enable the system to:
- Compare detections against the original query image, not just vehicle-to-vehicle
- Filter correlations by job, so searches don't contaminate each other
- Link tracks back to their originating search job for frontend display

Run once with:
    python migrate_add_query_tracking.py
"""
import sqlite3
import os

DB_PATH = os.path.join('instance', 'system.sqlite')

def migrate(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("Starting migration: adding query_embedding and job_id fields...")

    try:
        conn.execute("BEGIN")

        # Check if query_embedding already exists
        cur.execute("PRAGMA table_info(vehicle_detections)")
        columns = [row[1] for row in cur.fetchall()]
        
        has_query_embedding = 'query_embedding' in columns
        has_job_id_vd = 'job_id' in columns
        
        if has_query_embedding and has_job_id_vd:
            print("vehicle_detections already has query_embedding and job_id. Checking vehicle_tracks...")
        else:
            print("Migrating vehicle_detections table...")
            
            # Rename existing table
            conn.execute("ALTER TABLE vehicle_detections RENAME TO vehicle_detections_old")

            # Create new table with query_embedding and job_id
            conn.execute("""
                CREATE TABLE vehicle_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    box_x1 INTEGER NOT NULL,
                    box_y1 INTEGER NOT NULL,
                    box_x2 INTEGER NOT NULL,
                    box_y2 INTEGER NOT NULL,
                    box_area INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    query_embedding BLOB,
                    match_score REAL,
                    job_id TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (detection_id) REFERENCES detections(id),
                    FOREIGN KEY (camera_id) REFERENCES cameras(id),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)

            # Copy data from old table (job_id will be NULL for existing detections)
            conn.execute("""
                INSERT INTO vehicle_detections 
                (id, detection_id, camera_id, timestamp, box_x1, box_y1, box_x2, box_y2, 
                 box_area, embedding, match_score, created_at)
                SELECT 
                    id, detection_id, camera_id, timestamp, box_x1, box_y1, box_x2, box_y2, 
                    box_area, embedding, match_score, created_at
                FROM vehicle_detections_old
            """)

            # Drop old table
            conn.execute("DROP TABLE vehicle_detections_old")
            print("vehicle_detections migrated successfully.")

        # Check if vehicle_tracks already has job_id
        cur.execute("PRAGMA table_info(vehicle_tracks)")
        columns = [row[1] for row in cur.fetchall()]
        
        has_job_id_vt = 'job_id' in columns
        
        if not has_job_id_vt:
            print("Migrating vehicle_tracks table...")
            
            # Rename existing table
            conn.execute("ALTER TABLE vehicle_tracks RENAME TO vehicle_tracks_old")

            # Create new table with job_id
            conn.execute("""
                CREATE TABLE vehicle_tracks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    first_camera_id TEXT NOT NULL,
                    last_camera_id TEXT NOT NULL,
                    vehicle_count INTEGER DEFAULT 1,
                    FOREIGN KEY (first_camera_id) REFERENCES cameras(id),
                    FOREIGN KEY (last_camera_id) REFERENCES cameras(id),
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)

            # Copy data from old table (job_id will be NULL for existing tracks)
            conn.execute("""
                INSERT INTO vehicle_tracks 
                (id, first_seen, last_seen, first_camera_id, last_camera_id, vehicle_count)
                SELECT 
                    id, first_seen, last_seen, first_camera_id, last_camera_id, vehicle_count
                FROM vehicle_tracks_old
            """)

            # Drop old table
            conn.execute("DROP TABLE vehicle_tracks_old")
            print("vehicle_tracks migrated successfully.")
        else:
            print("vehicle_tracks already has job_id field.")

        conn.commit()
        print("Migration completed successfully!")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
    else:
        migrate(DB_PATH)
