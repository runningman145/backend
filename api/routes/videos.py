"""
Video management endpoints.
Handles video registration, retrieval, and batch processing.
"""
import os
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from ..db import get_db
from ..video_management import (
    register_video, get_video_metadata, get_camera_videos,
    get_unprocessed_videos, load_video_data
)

bp = Blueprint('videos', __name__, url_prefix='/videos')

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/register', methods=['POST'])
def register_video_route():
    """Register and upload a new video in the system."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get required form parameters
        camera_name = request.form.get('camera_name')
        
        if not camera_name:
            return jsonify({'error': 'camera_name is required'}), 400
        
        # Look up camera by name
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE name = ?', (camera_name,)).fetchone()
        if camera is None:
            return jsonify({'error': f"Camera '{camera_name}' not found"}), 404
        
        camera_id = camera['id']
        
        # Parse captured_at if provided, otherwise use current time
        captured_at_str = request.form.get('captured_at')
        if captured_at_str:
            try:
                captured_at = datetime.fromisoformat(captured_at_str)
            except ValueError:
                return jsonify({'error': 'Invalid captured_at format, use ISO format (e.g., 2026-04-20T14:20:00)'}), 400
        else:
            captured_at = datetime.now()
        
        # Save file to disk
        uploads_dir = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Create safe filename
        original_filename = secure_filename(file.filename)
        filename = f"{captured_at.strftime('%Y%m%d_%H%M%S')}_{original_filename}"
        storage_path = os.path.join(uploads_dir, filename)
        
        file.save(storage_path)
        
        # Get file size
        size_bytes = os.path.getsize(storage_path)
        
        # Register video in database
        video_id = register_video(
            camera_id=camera_id,
            filename=filename,
            storage_path=storage_path,
            captured_at=captured_at,
            size_bytes=size_bytes,
            duration_seconds=request.form.get('duration_seconds', type=float)
        )
        
        return jsonify({
            'video_id': video_id,
            'filename': filename,
            'size_bytes': size_bytes,
            'message': 'Video uploaded and registered successfully'
        }), 201
    
    except Exception as e:
        current_app.logger.error(f"Error registering video: {str(e)}")
        return jsonify({'error': f'Failed to register video: {str(e)}'}), 500


@bp.route('/by-camera/<camera_id>', methods=['GET'])
def list_camera_videos(camera_id):
    """List all videos from a specific camera."""
    try:
        limit = min(request.args.get('limit', 100, type=int), 1000)
        offset = request.args.get('offset', 0, type=int)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format'}), 400
        
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format'}), 400
        
        videos = get_camera_videos(
            camera_id, 
            limit=limit, 
            offset=offset,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'camera_id': camera_id,
            'videos': videos,
            'count': len(videos),
            'limit': limit,
            'offset': offset
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing camera videos: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500


@bp.route('/<video_id>/metadata', methods=['GET'])
def get_video_metadata_route(video_id):
    """Get metadata for a specific video."""
    try:
        video = get_video_metadata(video_id)
        if video is None:
            return jsonify({'error': 'Video not found'}), 404
        
        return jsonify(video), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting video metadata: {str(e)}")
        return jsonify({'error': f'Failed to get metadata: {str(e)}'}), 500


@bp.route('/unprocessed/list', methods=['GET'])
def list_unprocessed_videos_route():
    """List videos that haven't been processed yet."""
    try:
        camera_id = request.args.get('camera_id')
        limit = min(request.args.get('limit', 50, type=int), 500)
        
        videos = get_unprocessed_videos(camera_id=camera_id, limit=limit)
        
        return jsonify({
            'videos': videos,
            'count': len(videos),
            'filter_camera': camera_id
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing unprocessed videos: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500
