"""
Job model and management functions.
Handles job creation, status tracking, and database operations.
"""
import json
import uuid
from datetime import datetime
from flask import current_app
from ..db import get_db


def create_job(camera_id, detection_id, video_filename, query_image_filename, threshold=40, frame_skip=15):
    """
    Create a new job in the queue.
    
    Args:
        camera_id: UUID of the camera
        detection_id: Detection ID to associate with this job
        video_filename: Filename of the video to process
        query_image_filename: Filename of the query image
        threshold: Similarity threshold for matches
        frame_skip: Number of frames to skip
    
    Returns:
        job_id (UUID string)
    """
    job_id = str(uuid.uuid4())
    db = get_db()
    
    db.execute(
        'INSERT INTO jobs (id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip, status) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (job_id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip, 'pending')
    )
    db.commit()
    
    return job_id


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
        '       j.query_image_filename, c.name as camera_name '
        'FROM jobs j '
        'LEFT JOIN cameras c ON j.camera_id = c.id '
        'WHERE j.id = ?',
        (job_id,)
    ).fetchone()
    
    if job is None:
        return None
    
    result = {
        'id': job['id'],
        'status': job['status'],
        'created_at': job['created_at'],
        'started_at': job['started_at'],
        'completed_at': job['completed_at'],
        'camera_name': job['camera_name'],
        'query_image_filename': job['query_image_filename'],
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
