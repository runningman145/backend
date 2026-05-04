"""
Background job worker.
Processes video detection jobs from the queue.
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime
from flask import current_app
from ..db import get_db
from ..ml.loader import get_models
from ..ml.inference import extract_embedding, process_video_data
from .models import update_job_status


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
    
    # Get videos for the specified camera, date, and time range.
    # REPLACE(captured_at, 'T', ' ') normalises ISO-8601 'T'-separator
    # timestamps (stored by older code) so they compare correctly with
    # the space-separated datetime strings we build from job_date/time.
    query = '''
        SELECT id, filename FROM videos
        WHERE camera_id = ?
        AND REPLACE(captured_at, 'T', ' ') >= ?
        AND REPLACE(captured_at, 'T', ' ') <= ?
        ORDER BY captured_at
    '''
    videos = db.execute(query, (camera_id, start_datetime, end_datetime)).fetchall()
    
    if not videos:
        # Log more details for debugging
        all_videos = db.execute(
            'SELECT filename, REPLACE(captured_at, \'T\', \' \') as captured_at FROM videos WHERE camera_id = ? ORDER BY captured_at DESC LIMIT 5',
            (camera_id,)
        ).fetchall()
        current_app.logger.warning(f"Looking for videos between {start_datetime} and {end_datetime}")
        current_app.logger.warning(f"Recent videos for camera {camera_id}: {[dict(v) for v in all_videos]}")
        raise ValueError(f"No videos found for camera {camera_id} between {start_datetime} and {end_datetime}")
    
    all_results = []
    
    # Process each query image against all videos
    for query_image_filename in query_images:
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
        for video in videos:
            # Check for cancellation before each video
            if _is_cancelled(job_id):
                current_app.logger.info(f"Job {job_id} cancelled during batch processing")
                return

            video_path = os.path.join(upload_folder, video['filename'])

            if not os.path.exists(video_path):
                current_app.logger.warning(f"Video file not found: {video_path}")
                continue

            try:
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


def update_job_progress(job_id, frames_total, frames_processed):
    """
    Update job progress during processing.
    
    Args:
        job_id: UUID of the job
        frames_total: Total number of frames in video
        frames_processed: Number of frames processed so far
    """
    progress_percent = (frames_processed / frames_total * 100) if frames_total > 0 else 0
    
    progress_data = json.dumps({
        'frames_total': frames_total,
        'frames_processed': frames_processed,
        'progress_percent': round(progress_percent, 1),
    })
    
    update_job_status(job_id, 'processing', result_data=progress_data)
