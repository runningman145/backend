"""
Job queue endpoints.
Provides endpoints for queuing and monitoring video processing jobs.
"""
import json
import csv
from io import StringIO, BytesIO
from flask import Blueprint, jsonify, request, current_app, send_file
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
        # we would also want to select a time range and and a camera location,
        # parameter to get from frontend, multiple images can have single job
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
        
        # TODO: return name of file being processed
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


@bp.route('/<job_id>/results/json', methods=['GET'])
def get_job_results_json(job_id):
    """
    Download raw results as JSON.
    
    Returns:
        JSON file with raw results from the completed job.
        Only available for completed jobs.
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': f"Results not available for job in '{job['status']}' status"
            }), 400
        
        db = get_db()
        result = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        
        if not result or not result['result_data']:
            return jsonify({'error': 'No results found for this job'}), 404
        
        result_data = json.loads(result['result_data'])
        
        # Create BytesIO object for file download
        json_bytes = BytesIO(
            json.dumps(result_data, indent=2).encode('utf-8')
        )
        json_bytes.seek(0)
        
        return send_file(
            json_bytes,
            mimetype='application/json',
            as_attachment=True,
            download_name=f'job_{job_id}_results.json'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error downloading job results as JSON: {str(e)}")
        return jsonify({'error': f'Failed to download results: {str(e)}'}), 500


@bp.route('/<job_id>/results/csv', methods=['GET'])
def get_job_results_csv(job_id):
    """
    Download raw results as CSV.
    
    Returns:
        CSV file with results from the completed job.
        Only available for completed jobs.
        Flattens nested JSON structures.
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] != 'completed':
            return jsonify({
                'error': f"Results not available for job in '{job['status']}' status"
            }), 400
        
        db = get_db()
        result = db.execute(
            'SELECT result_data FROM jobs WHERE id = ?',
            (job_id,)
        ).fetchone()
        
        if not result or not result['result_data']:
            return jsonify({'error': 'No results found for this job'}), 404
        
        result_data = json.loads(result['result_data'])
        
        # Helper function to flatten nested structures
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary."""
            items = []
            if isinstance(d, dict):
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list):
                        # Convert lists to JSON string
                        items.append((new_key, json.dumps(v)))
                    else:
                        items.append((new_key, v))
            elif isinstance(d, list):
                # If root is a list, process each item
                for i, item in enumerate(d):
                    flattened = flatten_dict(item, '', sep=sep)
                    if i == 0:
                        items.extend(flattened.items())
                    else:
                        break
            return dict(items)
        
        # Handle both dict and list results
        if isinstance(result_data, list):
            # List of detections/matches
            rows = [flatten_dict(item) for item in result_data]
        else:
            # Single result object
            rows = [flatten_dict(result_data)]
        
        if not rows:
            return jsonify({'error': 'No data to export'}), 400
        
        # Create CSV in memory
        csv_buffer = StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(rows)
        
        csv_bytes = BytesIO(csv_buffer.getvalue().encode('utf-8'))
        csv_bytes.seek(0)
        
        return send_file(
            csv_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'job_{job_id}_results.csv'
        )
    
    except Exception as e:
        current_app.logger.error(f"Error downloading job results as CSV: {str(e)}")
        return jsonify({'error': f'Failed to download results: {str(e)}'}), 500

