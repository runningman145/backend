-- a table for cameras in the system

DROP TABLE IF EXISTS detection_matches;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS cameras;

CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude TEXT NOT NULL,
    longitude TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'offline' CHECK(status IN ('online', 'offline', 'inactive')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- a table for car detections recorded by the ML model
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    captured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

-- a table for ReID matching results
CREATE TABLE detection_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    timestamp REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detection_id) REFERENCES detections(id)
);

-- a table for individual vehicle detections with embeddings (for cross-camera tracking)
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
    match_score REAL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (detection_id) REFERENCES detections(id),
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

-- a table for vehicle tracks (grouped detections across cameras)
CREATE TABLE vehicle_tracks (
    id TEXT PRIMARY KEY,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_camera_id TEXT NOT NULL,
    last_camera_id TEXT NOT NULL,
    vehicle_count INTEGER DEFAULT 1,
    FOREIGN KEY (first_camera_id) REFERENCES cameras(id),
    FOREIGN KEY (last_camera_id) REFERENCES cameras(id)
);

-- junction table: vehicle_detections <-> vehicle_tracks
CREATE TABLE track_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    vehicle_detection_id INTEGER NOT NULL,
    UNIQUE(track_id, vehicle_detection_id),
    FOREIGN KEY (track_id) REFERENCES vehicle_tracks(id),
    FOREIGN KEY (vehicle_detection_id) REFERENCES vehicle_detections(id)
);

-- a table for background job queue (video processing jobs)
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
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
);

-- a table for linking multiple query images to a single job
CREATE TABLE job_query_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    query_image_filename TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id),
    UNIQUE(job_id, query_image_filename)
);

-- a table for videos linked to cameras
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    size_bytes INTEGER,
    duration_seconds REAL,
    captured_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed INTEGER DEFAULT 0,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

-- index for faster queries by camera and date
CREATE INDEX idx_videos_camera_captured ON videos(camera_id, captured_at);
