# Backend Explained (Beginner Friendly)

This file explains the backend like you are new to backend development and Flask.

I will use this order for each major piece:
- **What it is**
- **Why it exists**
- **How it works**

---

## 1) Big Picture: What This Backend Does

This backend is a **Flask API server** for a vehicle detection/tracking system.

It does four main jobs:
- Manage cameras and their metadata (name, location, status)
- Accept uploaded images/videos
- Run ML-based matching in background jobs (YOLO + ReID)
- Store and serve tracking/reporting data through HTTP endpoints

In simple terms:
1. Frontend sends requests.
2. Flask routes receive them.
3. Data is validated and saved in SQLite.
4. Heavy ML work is put in background worker threads.
5. Results are saved and later fetched by frontend.

---

## 2) Project Structure (Mental Map)

Main backend package is `backend/api/`.

- `api/__init__.py` -> app factory (`create_app`) and startup wiring
- `api/db.py` -> database connection lifecycle
- `api/schema.sql` -> tables and relationships
- `api/routes/` -> REST endpoints grouped by feature
- `api/jobs/` -> queue and worker logic for async processing
- `api/ml/` -> model loading + inference helpers
- `api/tracking/` -> cross-camera correlation and track APIs
- `api/video_management.py` -> helper layer for video metadata/data operations

Think of this as:
- **Routes = doors in**
- **DB = memory**
- **Jobs/ML = factory machines**
- **Tracking/Reports = intelligence + outputs**

---

## 3) App Startup and Core Architecture

### 3.1 `create_app` in `api/__init__.py`

**What**
- Central function that builds the Flask app object.

**Why**
- App factory pattern keeps setup clean, testable, and configurable.

**How**
- Creates `Flask(...)`
- Loads config (default + env + optional test config)
- Ensures `instance/` folder exists
- Initializes extensions (`CORS`)
- Initializes DB hooks/CLI commands
- Registers all blueprints (route modules)
- On first incoming request, lazily:
  - loads ML models
  - starts background job queue workers

Important detail:
- The first request triggers expensive setup (`before_request` guarded by `models_loaded`), so startup looks fast but first request does initialization.

---

## 4) Database Layer

### 4.1 `api/db.py`

**What**
- Utility functions for SQLite connection handling.

**Why**
- Each request should reuse one connection and close it safely at request end.

**How**
- `get_db()`:
  - stores connection in Flask `g` (request-local storage)
  - returns `sqlite3.Row` rows (dictionary-like access)
- `close_db()`:
  - called at request teardown to close connection
- `init_db()`:
  - executes SQL from `schema.sql`
- `init-db` CLI command:
  - `flask --app api init-db`

### 4.2 Schema in `api/schema.sql`

Main tables and roles:
- `cameras`: camera registry
- `detections`: high-level detection events
- `detection_matches`: legacy match rows
- `vehicle_detections`: per-vehicle box + embedding
- `vehicle_tracks`: grouped journey across cameras
- `track_detections`: many-to-many link (track <-> vehicle detection)
- `jobs`: async processing lifecycle (`pending/processing/completed/failed`)
- `job_query_images`: supports multiple query images per batch job
- `videos`: uploaded/registered video metadata

Why this design:
- Split raw detection storage from correlation/tracking.
- Keep heavy ML outputs queryable over time.
- Keep async state durable in DB (jobs survive server restarts better than in-memory queues).

---

## 5) Extensions

### `api/extensions.py`

**What**
- Initializes CORS.

**Why**
- Browser frontend and backend are often on different origins (ports/domains). CORS allows frontend requests.

**How**
- `cors = CORS(app)` in one central place.

---

## 6) Route Registration and Endpoint Modules

### 6.1 Route registration
- `api/routes/__init__.py` registers:
  - health
  - cameras
  - detections
  - uploads
  - videos
  - jobs
  - reports
- `api/__init__.py` additionally registers:
  - `api/ml/routes.py` (`/model/...`)
  - `api/tracking/routes.py` (`/tracking/...`)

---

## 7) Feature-by-Feature Breakdown (What / Why / How)

## 7.1 Health Routes (`/health`)

**What**
- Endpoints to check if service and DB are alive/ready.

**Why**
- Useful for frontend checks, deployment probes, and debugging.

**How**
- `/health` and `/health/live`: simple success responses
- `/health/ready`: runs `SELECT 1` against DB
- `/health/status`: includes DB counts for cameras/detections

---

## 7.2 Camera Routes (`/cameras`)

**What**
- CRUD APIs for camera records + camera-scoped videos list.

**Why**
- Cameras are core entities used by detections, videos, tracking, reports.

**How**
- `POST /cameras`: validates required fields (`name`, `latitude`, `longitude`), generates UUID, inserts row.
- `GET /cameras`: lists all cameras.
- `GET /cameras/<id>`: fetch single camera.
- `PUT /cameras/<id>`: partial updates for allowed fields.
- `DELETE /cameras/<id>`: removes camera and related detections.
- `GET /cameras/<id>/videos`: paginated video list for that camera.

---

## 7.3 Detection Routes (`/detections`)

**What**
- Manage detection events, stats, and PDF export.

**Why**
- Detections are central evidence objects tied to cameras and matches.

**How**
- `GET /detections`: paginated and filterable list.
- `GET /detections/<id>`: full details + associated matches.
- `POST /detections`: create detection (requires valid `camera_id`).
- `PUT /detections/<id>`: update selected fields (`track_id`, `match_score`, `query_embedding`).
- `DELETE /detections/<id>`: remove detection + linked matches.
- `GET /detections/<id>/export-pdf`: builds PDF report in memory using ReportLab and returns file.
- `GET /detections/stats`: aggregate metrics by camera/date.

---

## 7.4 Upload Routes (`/uploads`)

**What**
- Handle incoming query media and create processing jobs.

**Why**
- User uploads query evidence; backend should store file and kick off matching.

**How**
- `POST /uploads`:
  - receives one file + `camera_name`
  - stores file in `instance/uploads/`
  - creates a detection
  - creates jobs against videos from other cameras
- `POST /uploads/batch`:
  - accepts up to 5 files + date/time range
  - creates one batch job with multiple query images
- `GET /uploads/<filename>`:
  - secure file download from uploads dir

---

## 7.5 Video Routes (`/videos`)

**What**
- Video registration/listing/metadata APIs.

**Why**
- Videos are large binary inputs; metadata lets system find and process the right clips later.

**How**
- `POST /videos/register`: validates upload + camera, saves file, stores metadata in `videos`.
- `GET /videos/by-camera/<camera_id>`: list videos with optional date filters.
- `GET /videos/<video_id>/metadata`: fetch one metadata record.
- `GET /videos/unprocessed/list`: list videos where `processed = 0`.

Uses helper module `api/video_management.py` for DB + storage logic.

---

## 7.6 Job Routes (`/jobs`)

**What**
- Queue jobs and query status/results.

**Why**
- ML processing is expensive; API should respond fast and process asynchronously.

**How**
- `POST /jobs`: create pending job row.
- `GET /jobs/<job_id>`: current status and result payload.
- `GET /jobs`: list jobs with filters.
- `GET /jobs/search/by-date-time`: query jobs by camera/date/time.
- `GET /jobs/<job_id>/results/json|csv`: export completed result payload.

---

## 7.7 Report Routes (`/reports`)

**What**
- Build court-ready PDF from officer-edited CSV and optionally verify digital signature.

**Why**
- Operational output for legal/investigative workflows.

**How**
- `POST /reports/generate`:
  - parse CSV metadata + sightings
  - keep only `include=true`
  - fetch camera coordinates
  - request Mapbox static map
  - generate signed multipage PDF with ReportLab
- `POST /reports/verify-signature`:
  - recompute HMAC signature/hash and compare

---

## 7.8 ML Routes (`/model`)

**What**
- Model health/config endpoints and older direct processing endpoints.

**Why**
- Debug model readiness and allow direct model operations if needed.

**How**
- `/model/status`: checks cached model state/device
- `/model/config`: returns inference config values
- `/model/extract-embedding`: image -> embedding
- `/model/compare-embeddings`: similarity score
- `/model/process-video`: queues video processing flow

---

## 7.9 Tracking Routes (`/tracking`)

**What**
- Read and configure cross-camera vehicle tracks.

**Why**
- A single vehicle can appear across multiple cameras; tracks connect those events.

**How**
- `GET /tracking/tracks`: paginated summary list
- `GET /tracking/tracks/<track_id>`: full track details
- `GET /tracking/cameras/<camera_id>/tracks`: tracks touching one camera
- `GET/PUT /tracking/config`: inspect/update correlation thresholds and filters

---

## 8) Job Queue and Worker Architecture

### 8.1 Queue (`api/jobs/queue.py`)

**What**
- A DB-polled, thread-based worker queue.

**Why**
- Prevent long API blocking; process jobs in background.

**How**
- `JobQueue.start(app)` starts worker daemon threads.
- Each worker loop:
  - fetches oldest `pending` job from DB
  - calls `process_job(job)`
  - sleeps briefly if no job
- Singleton accessor `get_job_queue()` ensures one queue instance.

### 8.2 Worker (`api/jobs/worker.py`)

**What**
- Executes one job end-to-end.

**Why**
- Encapsulate heavy ML logic and DB result writes.

**How**
- Marks job `processing`
- Loads cached models
- Branches:
  - single job (`process_single_job`)
  - batch job (`process_batch_job`)
- Calls inference helpers for video processing/matching
- Saves results JSON
- Marks `completed` or `failed`

---

## 9) ML Pipeline (YOLO + ReID)

## 9.1 Loader (`api/ml/loader.py`)

**What**
- Loads and caches models and preprocessing transforms.

**Why**
- Loading models per request is too expensive.

**How**
- Detects device (`cuda`, `mps`, `cpu`)
- Loads YOLO (`yolov8n.pt`)
- Loads ReID model (`resnet50` with final layer removed)
- Applies optional custom ReID weights via `REID_MODEL_PATH`
- Stores in module cache for reuse

## 9.2 Inference (`api/ml/inference.py`)

**What**
- Core detection + matching routines.

**Why**
- Keep algorithmic logic separate from HTTP route handlers.

**How**
- `extract_embedding(...)`: image -> normalized feature vector
- `process_video_data(...)`:
  - write bytes to temp file
  - read frames with OpenCV
  - skip frames based on `frame_skip`
  - run YOLO vehicle detection
  - crop each vehicle, extract embedding
  - compare to query embedding using cosine similarity
  - if above threshold, add result
  - optionally store vehicle detections + correlate tracks
  - cleanup temp file

---

## 10) Cross-Camera Correlation Logic

### `api/tracking/correlate.py`

**What**
- Links new vehicle detections to existing tracks.

**Why**
- Convert isolated detections into trajectories across camera network.

**How**
- For each new vehicle detection:
  - query recent detections from other cameras (time-windowed)
  - compare embeddings (cosine similarity)
  - optional spatial filter using camera GPS + max distance
  - if good match:
    - add to existing track, or create new track from pair
  - if no match:
    - create single-vehicle track

Tunable config:
- embedding threshold
- time window
- max camera distance
- spatial filter toggle

---

## 11) End-to-End Request Flows

## 11.1 Add a camera
1. Frontend `POST /cameras`
2. Route validates fields
3. DB insert in `cameras`
4. Respond with created camera JSON

## 11.2 Upload query image and start matching
1. Frontend `POST /uploads`
2. Backend saves image file
3. Creates `detections` row
4. Creates one or more `jobs` rows
5. Worker picks pending jobs
6. ML inference processes videos
7. Results saved in `jobs.result_data` (+ optional tracking tables)
8. Frontend polls `GET /jobs/<id>`

## 11.3 Generate court report
1. Frontend uploads edited CSV to `POST /reports/generate`
2. Backend parses and filters sightings
3. Loads camera coordinates from DB
4. Generates route map via Mapbox
5. Creates signed PDF and returns download

---

## 12) Why This Architecture Works

- **Separation of concerns**: routes vs business logic vs ML vs tracking
- **Async processing**: avoids request timeouts for heavy work
- **Durable state**: jobs and artifacts tracked in DB
- **Extensibility**: easy to add new routes and helpers by module
- **Operational visibility**: health checks + status endpoints

---

## 13) Important Beginner Concepts in This Codebase

- **Blueprint**: Flask way to group routes by feature.
- **App context / request context**: Flask’s scoped globals (`current_app`, `g`).
- **Factory pattern**: `create_app` builds configured app instance.
- **Async by workers**: not Celery here; custom thread workers poll DB.
- **Embedding**: numeric fingerprint of an image crop.
- **ReID**: compare embeddings to tell whether vehicles likely match.
- **Track**: a grouped timeline of linked detections across cameras.

---

## 14) Practical "How to Read This Backend" Path

If you want to learn by reading code in the best order:
1. `api/__init__.py`
2. `api/routes/__init__.py`
3. `api/db.py` + `api/schema.sql`
4. `api/routes/cameras.py` (simple CRUD)
5. `api/routes/uploads.py` + `api/jobs/models.py`
6. `api/jobs/queue.py` + `api/jobs/worker.py`
7. `api/ml/loader.py` + `api/ml/inference.py`
8. `api/tracking/store.py` + `api/tracking/correlate.py` + `api/tracking/routes.py`
9. `api/routes/reports.py`

This progression goes from easiest to most advanced.

---

## 15) Notes You Should Keep in Mind

- The backend is feature-rich, but not yet heavily abstracted into service classes; logic is mostly function/module-based.
- SQLite is simple and great for dev/smaller deployments, but high-scale production usually migrates to PostgreSQL.
- Worker threads are lightweight and easy to run, but robust distributed queues (Celery/RQ) are often used at larger scale.

---

## 16) One-Sentence Summary

This backend is a modular Flask API that combines camera/video CRUD, async ML processing, cross-camera correlation, and report generation into one SQLite-backed service.

