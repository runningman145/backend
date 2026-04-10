"""
API endpoints for vehicle tracking and cross-camera correlation.
"""
from flask import Blueprint, jsonify, request
from . import vehicle_tracking

bp = Blueprint('tracking_api', __name__, url_prefix='/tracking')


@bp.route('/tracks', methods=['GET'])
def list_tracks():
    """
    Get all vehicle tracks with pagination.
    
    Query parameters:
        limit: Number of tracks to return (default 100)
        offset: Pagination offset (default 0)
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate parameters
        limit = max(1, min(limit, 500))  # Cap at 500
        offset = max(0, offset)
        
        tracks = vehicle_tracking.get_all_tracks(limit=limit, offset=offset)
        
        return jsonify({
            'tracks': tracks,
            'count': len(tracks),
            'limit': limit,
            'offset': offset,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get tracks: {str(e)}'}), 500


@bp.route('/tracks/<track_id>', methods=['GET'])
def get_track(track_id):
    """
    Get full details of a vehicle track including all cameras and timestamps.
    """
    try:
        track = vehicle_tracking.get_vehicle_track(track_id)
        
        if track is None:
            return jsonify({'error': 'Track not found'}), 404
        
        return jsonify(track), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get track: {str(e)}'}), 500


@bp.route('/cameras/<camera_id>/tracks', methods=['GET'])
def get_camera_tracks(camera_id):
    """
    Get all vehicle tracks that include detections from a specific camera.
    
    Query parameters:
        limit: Number of tracks to return (default 50)
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        limit = max(1, min(limit, 200))  # Cap at 200
        
        tracks = vehicle_tracking.get_tracks_for_camera(camera_id, limit=limit)
        
        return jsonify({
            'camera_id': camera_id,
            'tracks': tracks,
            'count': len(tracks),
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get camera tracks: {str(e)}'}), 500


@bp.route('/config', methods=['GET'])
def get_tracking_config():
    """
    Get current cross-camera tracking configuration.
    """
    return jsonify({
        'config': vehicle_tracking.TRACKING_CONFIG,
    }), 200


@bp.route('/config', methods=['PUT'])
def update_tracking_config():
    """
    Update cross-camera tracking configuration.
    
    Request body should contain config parameters to update:
    {
        'EMBEDDING_SIMILARITY_THRESHOLD': 0.65,
        'TIME_WINDOW_SECONDS': 120,
        'MAX_DISTANCE_KM': 5,
        'SPATIAL_FILTER_ENABLED': true
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400
        
        # Update config (only for valid keys)
        valid_keys = {
            'EMBEDDING_SIMILARITY_THRESHOLD',
            'TIME_WINDOW_SECONDS',
            'MAX_DISTANCE_KM',
            'SPATIAL_FILTER_ENABLED',
        }
        
        for key, value in data.items():
            if key in valid_keys:
                vehicle_tracking.TRACKING_CONFIG[key] = value
        
        return jsonify({
            'message': 'Configuration updated',
            'config': vehicle_tracking.TRACKING_CONFIG,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to update config: {str(e)}'}), 500
