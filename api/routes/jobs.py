"""
Job queue endpoints.
Provides endpoints for queuing and monitoring video processing jobs.
"""
from flask import Blueprint, jsonify, request, current_app
from ..jobs import create_job, get_job_status
from ..db import get_db

bp = Blueprint('jobs', __name__, url_prefix='/jobs')


@bp.route('', methods=['POST'])
def queue_video_job():
    """
    Queue a video for background processing.
    
    Request body (JSON):
    {
        'video_data': '<base64-encoded video>',  // OR upload as multipart file 'video'
        'query_image': '<base64-encoded query image>',  // OR upload as multipart file 'query_image'
        'camera_id': '<camera ID>',
        'detection_id': '<optional detection_id to associate results>',
        'threshold': <optional similarity threshold, default 40>,
        'frame_skip': <optional frames to skip, default 15>
    }
    """
    try:
        # Validate camera_id
        camera_id = request.json.get('camera_id') if request.json else None
        if not camera_id:
            return jsonify({'error': 'camera_id is required'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (camera_id,)).fetchone()
        if camera is None:
            return jsonify({'error': f'Camera {camera_id} not found'}), 404
        
        # Get processing parameters
        threshold = request.json.get('threshold', 40) if request.json else 40
        frame_skip = request.json.get('frame_skip', 15) if request.json else 15
        detection_id = request.json.get('detection_id') if request.json else None
        video_filename = request.json.get('video_filename') if request.json else None
        query_image_filename = request.json.get('query_image_filename') if request.json else None
        
        if not video_filename or not query_image_filename:
            return jsonify({'error': 'video_filename and query_image_filename are required'}), 400
        
        # Queue job
        job_id = create_job(
            camera_id=camera_id,
            detection_id=detection_id,
            video_filename=video_filename,
            query_image_filename=query_image_filename,
            threshold=threshold,
            frame_skip=frame_skip
        )
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video queued for processing',
            'detection_id': detection_id,
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error queuing video: {str(e)}")
        return jsonify({'error': f'Failed to queue video: {str(e)}'}), 500


@bp.route('/<job_id>', methods=['GET'])
def get_job_status_route(job_id):
    """
    Get the status of a queued video processing job.
    
    Returns:
    {
        'id': '<job_id>',
        'status': 'pending|processing|completed|failed',
        'created_at': '<timestamp>',
        'started_at': '<timestamp or null>',
        'completed_at': '<timestamp or null>',
        'results': { ... } // only if completed
        'error': '<error message>' // only if failed
    }
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': f'Failed to get job status: {str(e)}'}), 500


@bp.route('', methods=['GET'])
def list_jobs():
    """
    Get all jobs with optional filtering.
    
    Query parameters:
        status: Filter by status (pending, processing, completed, failed)
        limit: Number of jobs to return (default 100, max 500)
        offset: Pagination offset (default 0)
    """
    try:
        status_filter = request.args.get('status')
        limit = min(request.args.get('limit', 100, type=int), 500)
        offset = request.args.get('offset', 0, type=int)
        
        db = get_db()
        
        query = 'SELECT id, status, created_at, started_at, completed_at FROM jobs'
        params = []
        
        if status_filter:
            query += ' WHERE status = ?'
            params.append(status_filter)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        jobs = db.execute(query, params).fetchall()
        
        return jsonify({
            'jobs': [dict(job) for job in jobs],
            'count': len(jobs),
            'limit': limit,
            'offset': offset,
            'filter_status': status_filter
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing jobs: {str(e)}")
        return jsonify({'error': f'Failed to list jobs: {str(e)}'}), 500
