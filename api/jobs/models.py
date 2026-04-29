"""
Job model and management functions.
Handles job creation, status tracking, and database operations.
"""
import json
import uuid
from datetime import datetime
from flask import current_app
from ..db import get_db


def create_job(camera_id, detection_id, video_filename, query_image_filename, threshold=40, frame_skip=15, 
               job_date=None, start_time=None, end_time=None):
    """
    Create a new job in the queue.
    
    Args:
        camera_id: UUID of the camera
        detection_id: Detection ID to associate with this job
        video_filename: Filename of the video to process
        query_image_filename: Filename of the query image (for backward compatibility)
        threshold: Similarity threshold for matches
        frame_skip: Number of frames to skip
        job_date: Optional date for the job (YYYY-MM-DD format)
        start_time: Optional start time for filtering videos (HH:MM:SS format)
        end_time: Optional end time for filtering videos (HH:MM:SS format)
    
    Returns:
        job_id (UUID string)
    """
    job_id = str(uuid.uuid4())
    db = get_db()
    
    db.execute(
        'INSERT INTO jobs (id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip, '
        'job_date, start_time, end_time, status) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (job_id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip,
         job_date, start_time, end_time, 'pending')
    )
    db.commit()
    
    return job_id


def create_batch_job(camera_id, query_image_filenames, threshold=40, frame_skip=15,
                    job_date=None, start_time=None, end_time=None):
    """
    Create a new batch job with multiple query images.
    
    Args:
        camera_id: UUID of the camera
        query_image_filenames: List of query image filenames
        threshold: Similarity threshold for matches
        frame_skip: Number of frames to skip
        job_date: Optional date for the job (YYYY-MM-DD format)
        start_time: Optional start time for filtering videos (HH:MM:SS format)
        end_time: Optional end time for filtering videos (HH:MM:SS format)
    
    Returns:
        job_id (UUID string)
    """
    job_id = str(uuid.uuid4())
    db = get_db()
    
    # Create the job entry
    db.execute(
        'INSERT INTO jobs (id, camera_id, threshold, frame_skip, job_date, start_time, end_time, status) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (job_id, camera_id, threshold, frame_skip, job_date, start_time, end_time, 'pending')
    )
    
    # Add all query images to the job_query_images table
    for filename in query_image_filenames:
        db.execute(
            'INSERT INTO job_query_images (job_id, query_image_filename) VALUES (?, ?)',
            (job_id, filename)
        )
    
    db.commit()
    
    return job_id


def get_job_query_images(job_id):
    """
    Get all query images associated with a job.
    
    Args:
        job_id: UUID of the job
    
    Returns:
        List of query image filenames
    """
    db = get_db()
    images = db.execute(
        'SELECT query_image_filename FROM job_query_images WHERE job_id = ? ORDER BY created_at',
        (job_id,)
    ).fetchall()
    
    return [img['query_image_filename'] for img in images]


def get_job_status(job_id):
    """
    Get job status and details.
    
    Args:
        job_id: UUID of the job
    
    Returns:
        Dict with job status or None if not found
    """
    db = get_db()
    job = db.execute(
        'SELECT j.id, j.status, j.result_data, j.error_message, j.created_at, j.started_at, j.completed_at, '
        '       j.query_image_filename, j.job_date, j.start_time, j.end_time, c.name as camera_name '
        'FROM jobs j '
        'LEFT JOIN cameras c ON j.camera_id = c.id '
        'WHERE j.id = ?',
        (job_id,)
    ).fetchone()
    
    if job is None:
        return None
    
    # Get all query images for this job
    query_images = get_job_query_images(job_id)
    
    result = {
        'id': job['id'],
        'status': job['status'],
        'created_at': job['created_at'],
        'started_at': job['started_at'],
        'completed_at': job['completed_at'],
        'camera_name': job['camera_name'],
        'query_image_filename': job['query_image_filename'],
        'query_images': query_images,
        'job_date': job['job_date'],
        'start_time': job['start_time'],
        'end_time': job['end_time'],
    }
    
    if job['result_data']:
        result_data = json.loads(job['result_data'])
        result.update(result_data)
    
    if job['error_message']:
        result['error'] = job['error_message']
    
    return result


def update_job_status(job_id, status, result_data=None, error_message=None):
    """
    Update job status in database.
    
    Args:
        job_id: UUID of the job
        status: New status ('pending', 'processing', 'completed', 'failed')
        result_data: Optional JSON string of results
        error_message: Optional error details
    """
    try:
        db = get_db()
        
        update_fields = ['status = ?']
        values = [status]
        
        if status == 'processing':
            update_fields.append('started_at = ?')
            values.append(datetime.utcnow().isoformat())
        elif status in ('completed', 'failed'):
            update_fields.append('completed_at = ?')
            values.append(datetime.utcnow().isoformat())
        
        if result_data is not None:
            update_fields.append('result_data = ?')
            values.append(result_data)
        
        if error_message is not None:
            update_fields.append('error_message = ?')
            values.append(error_message)
        
        values.append(job_id)
        
        set_clause = ', '.join(update_fields)
        db.execute(f'UPDATE jobs SET {set_clause} WHERE id = ?', values)
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error updating job status: {str(e)}")


def get_jobs_by_date_and_time(camera_id=None, job_date=None, start_time=None, end_time=None, limit=100, offset=0):
    """
    Get jobs filtered by date and time range.
    
    Args:
        camera_id: Optional camera ID to filter by
        job_date: Optional date in YYYY-MM-DD format
        start_time: Optional start time in HH:MM:SS format
        end_time: Optional end time in HH:MM:SS format
        limit: Maximum number of jobs to return
        offset: Pagination offset
    
    Returns:
        List of jobs matching the criteria
    """
    db = get_db()
    
    query = 'SELECT j.id, j.status, j.created_at, j.started_at, j.completed_at, j.job_date, j.start_time, j.end_time, c.name as camera_name FROM jobs j LEFT JOIN cameras c ON j.camera_id = c.id WHERE 1=1'
    params = []
    
    if camera_id:
        query += ' AND j.camera_id = ?'
        params.append(camera_id)
    
    if job_date:
        query += ' AND j.job_date = ?'
        params.append(job_date)
    
    if start_time:
        query += ' AND j.start_time = ?'
        params.append(start_time)
    
    if end_time:
        query += ' AND j.end_time = ?'
        params.append(end_time)
    
    query += ' ORDER BY j.created_at DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    jobs = db.execute(query, params).fetchall()
    
    return [dict(job) for job in jobs]

