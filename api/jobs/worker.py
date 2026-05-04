"""
Background job worker.
Processes video detection jobs from the queue.
"""
import os
import json
import cv2
from datetime import datetime, timedelta
from flask import current_app
from ..db import get_db
from ..ml.loader import get_models
from ..ml.inference import extract_embedding, process_video_data
from .models import update_job_status

# Define timezone offset for Uganda (EAT = UTC+3)
# Job times come in local timezone, but videos are stored in UTC
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
        # The queue already marked this job 'processing' atomically; bail out
        # immediately if it was cancelled between being claimed and now.
        if _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} was cancelled before processing started")
            return

        # Load models once
        models = get_models()
        yolo_model = models['yolo']
        reid_model = models['reid']
        transform_func = models['transform']
        device = models['device']

        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        db = get_db()

        # Check if this is a batch job (has job_date and time range)
        is_batch_job = job['job_date'] and job['start_time'] and job['end_time']

        if is_batch_job:
            # Batch job: process multiple query images against videos in time range
            process_batch_job(
                job, db, upload_folder, yolo_model, reid_model,
                transform_func, device
            )
        else:
            # Legacy single job: process single query image against single video
            process_single_job(
                job, db, upload_folder, yolo_model, reid_model,
                transform_func, device
            )

        # Only log success if the job wasn't cancelled mid-flight
        if not _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        error_msg = str(e)
        current_app.logger.error(f"Job {job_id} failed: {error_msg}")
        # Don't overwrite a cancelled status with 'failed'
        if not _is_cancelled(job_id):
            update_job_status(job_id, 'failed', error_message=error_msg)


def process_single_job(job, db, upload_folder, yolo_model, reid_model, transform_func, device):
    """Process a legacy single video job."""
    job_id = job['id']

    video_path = os.path.join(upload_folder, job['video_filename'])
    query_image_path = os.path.join(upload_folder, job['query_image_filename'])

    if not os.path.exists(video_path) or not os.path.exists(query_image_path):
        raise FileNotFoundError("Video or image file not found")

    query_image = cv2.imread(query_image_path)
    if query_image is None:
        raise ValueError("Could not read query image")

    query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_embedding = extract_embedding(
        query_image_rgb, reid_model, transform_func, device
    )

    # Pass video_path directly — no need to read the whole file into RAM
    def _on_video_progress(frames_read: int, frames_total: int):
        update_job_progress(
            job_id,
            frames_read=frames_read,
            frames_total=frames_total,
        )

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
    )

    # Store detection matches in database (frame_image is not a DB column – strip it)
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


def process_batch_job(job, db, upload_folder, yolo_model, reid_model, transform_func, device):
    """Process a batch job with multiple query images and time range filtering."""
    from .models import get_job_query_images
    
    job_id = job['id']
    camera_id = job['camera_id']
    job_date = job['job_date']
    start_time = job['start_time']
    end_time = job['end_time']
    threshold = job['threshold']
    frame_skip = job['frame_skip']
    
    # Get all query images for this batch job
    query_images = get_job_query_images(job_id)
    
    if not query_images:
        raise ValueError("Batch job has no query images")
    
    # Construct full datetime strings with seconds for proper comparison
    # If time doesn't have seconds, add :00
    if len(start_time.split(':')) == 2:
        start_time = f"{start_time}:00"
    if len(end_time.split(':')) == 2:
        end_time = f"{end_time}:00"
    
    start_datetime = f"{job_date} {start_time}"
    end_datetime = f"{job_date} {end_time}"
    
    # Convert local times (EAT) to UTC for database query
    # The job times are in local timezone (EAT, UTC+3), but captured_at in DB is UTC
    # So we need to subtract the timezone offset to get UTC equivalents
    try:
        start_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
        
        # Convert local times to UTC by subtracting the offset
        start_dt_utc = start_dt - TIMEZONE_OFFSET
        end_dt_utc = end_dt - TIMEZONE_OFFSET
        
        start_datetime_utc = start_dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        end_datetime_utc = end_dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        current_app.logger.info(f"Job {job_id}: Converting {start_datetime} EAT to {start_datetime_utc} UTC")
    except ValueError as e:
        current_app.logger.error(f"Job {job_id}: Failed to parse datetime: {e}")
        raise
    
    # Get videos for the specified camera, date, and time range.
    # Normalize captured_at to "YYYY-MM-DD HH:MM:SS" format by:
    # 1. REPLACE('T', ' ') to handle ISO-8601 format
    # 2. SUBSTR(..., 1, 19) to strip timezone info (e.g., +00:00)
    query = '''
        SELECT id, filename, SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) as captured_at_utc
        FROM videos
        WHERE camera_id = ?
        AND SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) >= ?
        AND SUBSTR(REPLACE(captured_at, 'T', ' '), 1, 19) <= ?
        ORDER BY captured_at
    '''
    videos = db.execute(query, (camera_id, start_datetime_utc, end_datetime_utc)).fetchall()
    
    if not videos:
        # Log more details for debugging
        all_videos = db.execute(
            'SELECT filename, SUBSTR(REPLACE(captured_at, \'T\', \' \'), 1, 19) as captured_at FROM videos WHERE camera_id = ? ORDER BY captured_at DESC LIMIT 5',
            (camera_id,)
        ).fetchall()
        current_app.logger.warning(f"Looking for videos between {start_datetime_utc} and {end_datetime_utc} (UTC)")
        current_app.logger.warning(f"Job requested: {start_datetime} to {end_datetime} (local EAT)")
        current_app.logger.warning(f"Recent videos for camera {camera_id}: {[dict(v) for v in all_videos]}")
        raise ValueError(f"No videos found for camera {camera_id} between {start_datetime_utc} and {end_datetime_utc} UTC")
    
    all_results = []
    num_queries = len(query_images)
    num_videos = len(videos)
    total_pairs = max(1, num_queries * num_videos)

    # Process each query image against all videos
    for qi, query_image_filename in enumerate(query_images):
        # Check for cancellation before each query image
        if _is_cancelled(job_id):
            current_app.logger.info(f"Job {job_id} cancelled during batch processing")
            return

        query_image_path = os.path.join(upload_folder, query_image_filename)

        if not os.path.exists(query_image_path):
            current_app.logger.warning(f"Query image not found: {query_image_path}")
            continue

        query_image = cv2.imread(query_image_path)
        if query_image is None:
            current_app.logger.warning(f"Could not read query image: {query_image_filename}")
            continue

        # Extract embedding for this query image
        query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_embedding = extract_embedding(
            query_image_rgb, reid_model, transform_func, device
        )

        # Process each video
        for vi, video in enumerate(videos):
            # Check for cancellation before each video
            if _is_cancelled(job_id):
                current_app.logger.info(f"Job {job_id} cancelled during batch processing")
                return

            video_path = os.path.join(upload_folder, video['filename'])

            if not os.path.exists(video_path):
                current_app.logger.warning(f"Video file not found: {video_path}")
                continue

            try:
                pair_flat = qi * num_videos + vi

                # -------------------------------------------------------- #
                # Trim within the selected video based on overlap           #
                # -------------------------------------------------------- #
                # videos.captured_at is treated as the video's start time in UTC.
                # Compute offsets directly against the requested job window.
                # This avoids relying on duration metadata (which may be missing
                # for some containers/codecs and would otherwise disable trimming).
                start_seconds = None
                end_seconds = None
                try:
                    video_start_utc = datetime.strptime(video['captured_at_utc'], '%Y-%m-%d %H:%M:%S')
                    job_start_utc = datetime.strptime(start_datetime_utc, '%Y-%m-%d %H:%M:%S')
                    job_end_utc = datetime.strptime(end_datetime_utc, '%Y-%m-%d %H:%M:%S')
                    start_seconds = max(0.0, (job_start_utc - video_start_utc).total_seconds())
                    end_seconds = (job_end_utc - video_start_utc).total_seconds()
                    # If the requested window is entirely before this video start,
                    # there is no overlap to process.
                    if end_seconds <= 0:
                        continue
                    # If clamping produced an empty interval, skip safely.
                    if end_seconds <= start_seconds:
                        continue
                except Exception:
                    # If overlap computation fails, fall back to processing full selected file.
                    start_seconds = None
                    end_seconds = None

                def _on_batch_video_progress(frames_read: int, frames_total: int, _pf=pair_flat):
                    if frames_total and frames_total > 0:
                        segment = frames_read / frames_total
                    else:
                        segment = min(0.95, frames_read / 30000.0)
                    overall = 100.0 * (_pf + segment) / total_pairs
                    overall = min(99.9, overall)
                    update_job_progress(
                        job_id,
                        frames_read=frames_read,
                        frames_total=frames_total,
                        progress_percent=overall,
                    )

                # Pass path directly — avoids loading the whole video into RAM
                results = process_video_data(
                    video_data=None,
                    yolo_model=yolo_model,
                    reid_model=reid_model,
                    transform_func=transform_func,
                    device=device,
                    query_embedding=query_embedding,
                    threshold=threshold,
                    frame_skip=frame_skip,
                    detection_id=None,
                    camera_id=camera_id,
                    video_path=video_path,
                    video_progress_callback=_on_batch_video_progress,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )

                # Add video and query image info to results
                for result in results:
                    result['query_image'] = query_image_filename
                    result['video_id'] = video['id']
                    result['video_filename'] = video['filename']

                all_results.extend(results)
            except Exception as e:
                current_app.logger.error(f"Error processing video {video['filename']}: {str(e)}")
                continue
    
    # Mark job as completed with all results
    result_data = json.dumps({
        'matches': all_results,
        'total_matches': len(all_results),
        'query_images_count': len(query_images),
        'videos_processed': len(videos),
    })
    update_job_status(job_id, 'completed', result_data=result_data)


def update_job_progress(job_id, frames_read, frames_total, progress_percent=None):
    """
    Persist in-progress video decode position for the Jobs UI.

    Only updates result_data; does not change status or started_at (safe to call often).

    Args:
        job_id: UUID of the job
        frames_read: Number of frames successfully decoded so far (video timeline)
        frames_total: Reported frame count from the container (0 if unknown)
        progress_percent: Optional precomputed overall percent (e.g. batch jobs spanning
            multiple videos). When omitted, percent is derived from frames_read / frames_total.
    """
    if progress_percent is None and frames_total and frames_total > 0:
        progress_percent = min(100.0, (frames_read / frames_total) * 100)

    progress_data = json.dumps({
        'frames_total': int(frames_total) if frames_total and frames_total > 0 else None,
        'frames_read': int(frames_read),
        'frames_processed': int(frames_read),  # legacy key for older clients
        'progress_percent': round(float(progress_percent), 1) if progress_percent is not None else None,
    })

    db = get_db()
    db.execute(
        'UPDATE jobs SET result_data = ? WHERE id = ? AND status = ?',
        (progress_data, job_id, 'processing'),
    )
    db.commit()
