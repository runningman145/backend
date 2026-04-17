"""
Video management endpoints.
Handles video registration, retrieval, and batch processing.
"""
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app
from ..db import get_db
from ..video_management import (
    register_video, get_video_metadata, get_camera_videos,
    get_unprocessed_videos, load_video_data
)

bp = Blueprint('videos', __name__, url_prefix='/videos')


@bp.route('/register', methods=['POST'])
def register_video_route():
    """Register a new video in the system."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        required = ['camera_id', 'filename', 'storage_path', 'captured_at']
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (data['camera_id'],)).fetchone()
        if camera is None:
            return jsonify({'error': f"Camera {data['camera_id']} not found"}), 404
        
        # Parse captured_at
        try:
            captured_at = datetime.fromisoformat(data['captured_at'])
        except ValueError:
            return jsonify({'error': 'Invalid captured_at format, use ISO format'}), 400
        
        video_id = register_video(
            camera_id=data['camera_id'],
            filename=data['filename'],
            storage_path=data['storage_path'],
            captured_at=captured_at,
            size_bytes=data.get('size_bytes'),
            duration_seconds=data.get('duration_seconds')
        )
        
        return jsonify({
            'video_id': video_id,
            'message': 'Video registered successfully'
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
