"""
Background job worker.
Processes video detection jobs from the queue.
"""
import json
import os
import cv2
from datetime import datetime, timedelta
from flask import current_app
from ..db import get_db
from ..ml.loader import get_models
from ..ml.inference import extract_embedding, process_video_data
from .models import update_job_status

# Define timezone offset for Uganda (EAT = UTC+3)
TIMEZONE_OFFSET = timedelta(hours=3)


def _is_cancelled(job_id):
    """Poll the DB to check if this job has been marked cancelled."""
    try:
        db = get_db()
        row = db.execute('SELECT status FROM jobs WHERE id = ?', (job_id,)).fetchone()
        return row is not None and row['status'] == 'cancelled'
    except Exception:
        return False


def process_job(job):
    """Process a single video detection job."""
    job_id = job['id']

    try:
        if _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} was cancelled before processing started")
            return

        models        = get_models()
        yolo_model    = models['yolo']
        reid_model    = models['reid']
        transform_func = models['transform']
        device        = models['device']

        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        db = get_db()

        is_batch_job = job['job_date'] and job['start_time'] and job['end_time']

        if is_batch_job:
            process_batch_job(
                job, db, upload_folder, yolo_model, reid_model, transform_func, device
            )
        else:
            process_single_job(
                job, db, upload_folder, yolo_model, reid_model, transform_func, device
            )

        if not _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        current_app.logger.error(f"Job {job_id} failed: {error_msg}")
        if not _is_cancelled(job_id):
            update_job_status(job_id, 'failed', error_message=error_msg)


def process_single_job(job, db, upload_folder, yolo_model, reid_model, transform_func, device):
    """Process a legacy single video job."""
    job_id = job['id']

    video_path       = os.path.join(upload_folder, job['video_filename'])
    query_image_path = os.path.join(upload_folder, job['query_image_filename'])

    if not os.path.exists(video_path) or not os.path.exists(query_image_path):
        raise FileNotFoundError("Video or image file not found")

    query_image = cv2.imread(query_image_path)
    if query_image is None:
        raise ValueError("Could not read query image")

    query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_embedding = extract_embedding(query_image_rgb, reid_model, transform_func, device)

    def _on_video_progress(frames_read: int, frames_total: int):
        update_job_progress(job_id, frames_read=frames_read, frames_total=frames_total)

    results = process_video_data(
        video_data=None,
        yolo_model=yolo_model,
        reid_model=reid_model,
        transform_func=transform_func,
        device=device,
        query_embedding=query_embedding,
        threshold=job['threshold'],
        frame_skip=job['frame_skip'],
        detection_id=job['detection_id'],
        camera_id=job['camera_id'],
        video_path=video_path,
        video_progress_callback=_on_video_progress,
        job_id=job_id,
    )

    for result in results:
        db.execute(
            'INSERT INTO detection_matches (detection_id, similarity_score, timestamp) VALUES (?, ?, ?)',
            (job['detection_id'], result['match_percent'], result['time'])
        )
    db.commit()

    result_data = json.dumps({
        'matches': results,
        'total_matches': len(results),
    })
    update_job_status(job_id, 'completed', result_data=result_data)


def _build_camera_list(job, db):
    """
    Return the list of camera entries to process for a batch job.

    Supports two formats:
      1. New:  `camera_ids` column contains JSON with per-camera time ranges.
      2. Old:  Only `camera_id` / `start_time` / `end_time` columns exist.

    Returns:
        list of dicts: [{camera_id, camera_name, start_time, end_time}, ...]
    """
    # Try the new multi-camera JSON column first
    raw_camera_ids = job['camera_ids'] if 'camera_ids' in job.keys() else None
    if raw_camera_ids:
        try:
            entries = json.loads(raw_camera_ids)
            if entries and isinstance(entries, list):
                return entries
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to the single-camera legacy format
    return [{
        'camera_id':   job['camera_id'],
        'camera_name': None,
        'start_time':  job['start_time'],
        'end_time':    job['end_time'],
    }]


def process_batch_job(job, db, upload_folder, yolo_model, reid_model, transform_func, device):
    """
    Process a batch job.

    Iterates over all cameras listed in `camera_ids` (or falls back to the
    single-camera legacy format) and processes every query image against
    every video clip that falls within each camera's time range.
    """
    from .models import get_job_query_images

    job_id     = job['id']
    job_date   = job['job_date']
    threshold  = job['threshold']
    frame_skip = job['frame_skip']

    # ------------------------------------------------------------------ #
    # 1. Resolve camera list                                               #
    # ------------------------------------------------------------------ #
    camera_entries = _build_camera_list(job, db)
    current_app.logger.info(
        f"Job {job_id}: processing {len(camera_entries)} camera(s)"
    )

    # ------------------------------------------------------------------ #
    # 2. Query images                                                      #
    # ------------------------------------------------------------------ #
    query_images = get_job_query_images(job_id)
    if not query_images:
        raise ValueError("Batch job has no query images")

    # ------------------------------------------------------------------ #
    # 3. Create a shared detection record (primary camera)                 #
    # ------------------------------------------------------------------ #
    detection_cursor = db.execute(
        'INSERT INTO detections (camera_id) VALUES (?)',
        (job['camera_id'],)
    )
    db.commit()
    batch_detection_id = detection_cursor.lastrowid

    # ------------------------------------------------------------------ #
    # 4. Pre-load query embeddings (extract once, reuse per video)         #
    # ------------------------------------------------------------------ #
    query_embeddings = []   # [(filename, embedding)]
    for query_image_filename in query_images:
        path = os.path.join(upload_folder, query_image_filename)
        if not os.path.exists(path):
            current_app.logger.warning(f"Query image not found: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            current_app.logger.warning(f"Could not read query image: {query_image_filename}")
            continue
        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = extract_embedding(img_rgb, reid_model, transform_func, device)
        query_embeddings.append((query_image_filename, embedding))

    if not query_embeddings:
        raise ValueError("No query images could be loaded for this batch job")

    # ------------------------------------------------------------------ #
    # 5. Count total work for overall progress                             #
    # ------------------------------------------------------------------ #
    # We'll count videos per camera below; initialise to 1 so we never /0
    total_pairs    = 1
    pairs_done     = 0
    all_results    = []

    # ------------------------------------------------------------------ #
    # 6. Main loop: cameras → videos → query images                        #
    # ------------------------------------------------------------------ #
    for cam_entry in camera_entries:
        if _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} cancelled during batch processing")
            return

        camera_id  = cam_entry['camera_id']
        start_time = cam_entry.get('start_time') or job['start_time']
        end_time   = cam_entry.get('end_time')   or job['end_time']

        # Normalise time strings
        if len(start_time.split(':')) == 2:
            start_time = f"{start_time}:00"
        if len(end_time.split(':')) == 2:
            end_time = f"{end_time}:00"

        start_datetime = f"{job_date} {start_time}"
        end_datetime   = f"{job_date} {end_time}"

        # Convert EAT → UTC
        try:
            start_dt_utc = (datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S') - TIMEZONE_OFFSET)
            end_dt_utc   = (datetime.strptime(end_datetime,   '%Y-%m-%d %H:%M:%S') - TIMEZONE_OFFSET)
            start_datetime_utc = start_dt_utc.strftime('%Y-%m-%d %H:%M:%S')
            end_datetime_utc   = end_dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            current_app.logger.error(f"Job {job_id}: datetime parse error for camera {camera_id}: {e}")
            continue

        current_app.logger.info(
            f"Job {job_id} / camera {camera_id}: "
            f"{start_datetime} → {end_datetime} EAT  "
            f"({start_datetime_utc} → {end_datetime_utc} UTC)"
        )

        # Find videos for this camera in the requested window
        videos = db.execute(
            '''SELECT id, filename,
                      SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) as captured_at_utc
               FROM videos
               WHERE camera_id = ?
               AND SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) >= ?
               AND SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) <= ?
               ORDER BY captured_at''',
            (camera_id, start_datetime_utc, end_datetime_utc)
        ).fetchall()

        if not videos:
            current_app.logger.warning(
                f"Job {job_id}: no videos found for camera {camera_id} "
                f"between {start_datetime_utc} and {end_datetime_utc} UTC"
            )
            continue

        # Accumulate total pairs for progress tracking
        total_pairs += len(query_embeddings) * len(videos)

        for qi, (query_image_filename, query_embedding) in enumerate(query_embeddings):
            if _is_cancelled(job_id):
                current_app.logger.info(f"Job {job_id} cancelled during batch processing")
                return

            for vi, video in enumerate(videos):
                if _is_cancelled(job_id):
                    return

                video_path = os.path.join(upload_folder, video['filename'])
                if not os.path.exists(video_path):
                    current_app.logger.warning(f"Video file not found: {video_path}")
                    pairs_done += 1
                    continue

                # Compute within-video start/end offsets
                start_seconds = None
                end_seconds   = None
                try:
                    video_start_utc = datetime.strptime(video['captured_at_utc'], '%Y-%m-%d %H:%M:%S')
                    job_start_utc   = datetime.strptime(start_datetime_utc, '%Y-%m-%d %H:%M:%S')
                    job_end_utc     = datetime.strptime(end_datetime_utc,   '%Y-%m-%d %H:%M:%S')
                    start_seconds   = max(0.0, (job_start_utc - video_start_utc).total_seconds())
                    end_seconds     = (job_end_utc - video_start_utc).total_seconds()
                    if end_seconds <= 0 or end_seconds <= start_seconds:
                        pairs_done += 1
                        continue
                except Exception:
                    start_seconds = None
                    end_seconds   = None

                pair_flat = pairs_done

                def _on_progress(frames_read: int, frames_total: int, _pf=pair_flat):
                    if frames_total and frames_total > 0:
                        segment = frames_read / frames_total
                    else:
                        segment = min(0.95, frames_read / 30000.0)
                    overall = 100.0 * (_pf + segment) / max(1, total_pairs)
                    overall = min(99.9, overall)
                    update_job_progress(
                        job_id,
                        frames_read=frames_read,
                        frames_total=frames_total,
                        progress_percent=overall,
                    )

                try:
                    results = process_video_data(
                        video_data=None,
                        yolo_model=yolo_model,
                        reid_model=reid_model,
                        transform_func=transform_func,
                        device=device,
                        query_embedding=query_embedding,
                        threshold=threshold,
                        frame_skip=frame_skip,
                        detection_id=batch_detection_id,
                        camera_id=camera_id,
                        video_path=video_path,
                        video_progress_callback=_on_progress,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        job_id=job_id,
                    )

                    for result in results:
                        result['query_image']    = query_image_filename
                        result['video_id']       = video['id']
                        result['video_filename'] = video['filename']
                        result['camera_id']      = camera_id
                        result['camera_name']    = cam_entry.get('camera_name')

                    all_results.extend(results)
                except Exception as e:
                    current_app.logger.error(
                        f"Error processing video {video['filename']} "
                        f"for camera {camera_id}: {str(e)}"
                    )

                pairs_done += 1

    # ------------------------------------------------------------------ #
    # 7. Save final results                                                #
    # ------------------------------------------------------------------ #
    result_data = json.dumps({
        'matches':             all_results,
        'total_matches':       len(all_results),
        'query_images_count':  len(query_embeddings),
        'cameras_processed':   len(camera_entries),
        'detection_id':        batch_detection_id,
    })
    update_job_status(job_id, 'completed', result_data=result_data)


def update_job_progress(job_id, frames_read, frames_total, progress_percent=None):
    """
    Persist in-progress video decode position for the Jobs UI.
    """
    if progress_percent is None and frames_total and frames_total > 0:
        progress_percent = min(100.0, (frames_read / frames_total) * 100)

    progress_data = json.dumps({
        'frames_total':      int(frames_total) if frames_total and frames_total > 0 else None,
        'frames_read':       int(frames_read),
        'frames_processed':  int(frames_read),
        'progress_percent':  round(float(progress_percent), 1) if progress_percent is not None else None,
    })

    db = get_db()
    db.execute(
        'UPDATE jobs SET result_data = ? WHERE id = ? AND status = ?',
        (progress_data, job_id, 'processing'),
    )
    db.commit()