"""
Vehicle tracking endpoints.
Provides endpoints for querying vehicle tracks and cross-camera correlations.
"""
from datetime import datetime as _dt, timezone as _tz
from flask import Blueprint, jsonify, request
from ..db import get_db
from .store import store_vehicle_detection
from .correlate import TRACKING_CONFIG


def _fmt_detection_ts(ts, fallback):
    """
    Return a display timestamp for a vehicle_detection row.

    Batch jobs store absolute UTC epoch seconds (> 1e9).
    Single-video jobs store a video offset in seconds (small float).
    Returns an ISO-format string for epoch values, or falls back to the
    detection-session captured_at for legacy video-offset values.
    """
    try:
        t = float(ts)
        if t > 1_000_000_000:
            return _dt.fromtimestamp(t, tz=_tz.utc).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        pass
    return fallback

bp = Blueprint('tracking_api', __name__, url_prefix='/tracking')


def get_vehicle_track(track_id):
    """
    Get all detections in a track with camera info.
    
    Returns:
        Track details with all vehicle detections and sightings for map rendering
    """
    try:
        db = get_db()
        track = db.execute(
            '''SELECT vt.id, vt.job_id, vt.first_seen, vt.last_seen, 
                      COUNT(DISTINCT td.vehicle_detection_id) as detection_count,
                      COUNT(DISTINCT vd.camera_id) as camera_count
               FROM vehicle_tracks vt
               LEFT JOIN track_detections td ON vt.id = td.track_id
               LEFT JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               WHERE vt.id = ?
               GROUP BY vt.id''',
            (track_id,)
        ).fetchone()
        
        if not track:
            return None
        
        detections = db.execute(
            '''SELECT vd.id, vd.timestamp, vd.match_score, vd.box_x1, vd.box_y1, vd.box_x2, vd.box_y2, 
                      c.id as camera_id, c.name as camera_name, c.latitude, c.longitude, d.captured_at
               FROM track_detections td
               JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               JOIN cameras c ON vd.camera_id = c.id
               JOIN detections d ON vd.detection_id = d.id
               WHERE td.track_id = ?
               ORDER BY vd.timestamp ASC''',
            (track_id,)
        ).fetchall()
        
        return {
            'track_id': track['id'],
            'job_id': track['job_id'],
            'first_seen': track['first_seen'],
            'last_seen': track['last_seen'],
            'detection_count': track['detection_count'],
            'camera_count': track['camera_count'],
            'sightings': [
                {
                    'camera_id': d['camera_id'],
                    'camera_name': d['camera_name'],
                    'latitude': d['latitude'],
                    'longitude': d['longitude'],
                    # Use the stored absolute UTC timestamp when available
                    # (batch jobs); fall back to the session captured_at for
                    # legacy single-video jobs that store video offsets.
                    'timestamp': _fmt_detection_ts(d['timestamp'], d['captured_at']),
                    'confidence': d['match_score'],
                }
                for d in detections
            ],
            'detections': [
                {
                    'detection_id': d['id'],
                    'timestamp': d['timestamp'],
                    'match_score': d['match_score'],
                    'bbox': {'x1': d['box_x1'], 'y1': d['box_y1'], 'x2': d['box_x2'], 'y2': d['box_y2']},
                    'camera': {
                        'id': d['camera_id'],
                        'name': d['camera_name'],
                        'latitude': d['latitude'],
                        'longitude': d['longitude'],
                    },
                    'captured_at': d['captured_at'],
                }
                for d in detections
            ]
        }
    
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Error getting vehicle track: {str(e)}")
        return None


def get_all_tracks(limit=100, offset=0, job_id=None):
    """Get all vehicle tracks with summary info, optionally filtered by job_id."""
    try:
        db = get_db()
        
        if job_id:
            # Filter by specific job
            tracks = db.execute(
                '''SELECT vt.id, vt.job_id, vt.first_seen, vt.last_seen, 
                          COUNT(DISTINCT td.vehicle_detection_id) as detection_count,
                          COUNT(DISTINCT vd.camera_id) as camera_count
                   FROM vehicle_tracks vt
                   LEFT JOIN track_detections td ON vt.id = td.track_id
                   LEFT JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
                   WHERE vt.job_id = ?
                   GROUP BY vt.id
                   ORDER BY vt.last_seen DESC
                   LIMIT ? OFFSET ?''',
                (job_id, limit, offset)
            ).fetchall()
        else:
            # Get all tracks (only those with job_id set - filtered tracks)
            tracks = db.execute(
                '''SELECT vt.id, vt.job_id, vt.first_seen, vt.last_seen, 
                          COUNT(DISTINCT td.vehicle_detection_id) as detection_count,
                          COUNT(DISTINCT vd.camera_id) as camera_count
                   FROM vehicle_tracks vt
                   LEFT JOIN track_detections td ON vt.id = td.track_id
                   LEFT JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
                   WHERE vt.job_id IS NOT NULL
                   GROUP BY vt.id
                   ORDER BY vt.last_seen DESC
                   LIMIT ? OFFSET ?''',
                (limit, offset)
            ).fetchall()
        
        return [
            {
                'track_id': t['id'],
                'job_id': t['job_id'],
                'first_seen': t['first_seen'],
                'last_seen': t['last_seen'],
                'detection_count': t['detection_count'],
                'camera_count': t['camera_count'],
            }
            for t in tracks
        ]
    
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Error getting tracks: {str(e)}")
        return []


def get_tracks_for_camera(camera_id, limit=50):
    """Get all vehicle tracks that include detections from a specific camera."""
    try:
        db = get_db()
        tracks = db.execute(
            '''SELECT DISTINCT vt.id, vt.first_seen, vt.last_seen,
                      COUNT(DISTINCT td.vehicle_detection_id) as detection_count
               FROM vehicle_tracks vt
               JOIN track_detections td ON vt.id = td.track_id
               JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               WHERE vd.camera_id = ?
               GROUP BY vt.id
               ORDER BY vt.last_seen DESC
               LIMIT ?''',
            (camera_id, limit)
        ).fetchall()
        
        return [
            {
                'track_id': t['id'],
                'first_seen': t['first_seen'],
                'last_seen': t['last_seen'],
                'detection_count': t['detection_count'],
            }
            for t in tracks
        ]
    
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Error getting camera tracks: {str(e)}")
        return []


@bp.route('/tracks', methods=['GET'])
def list_tracks():
    """Get all vehicle tracks with pagination, optionally filtered by job_id."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        job_id = request.args.get('job_id', None, type=str)  # Optional job_id filter
        
        # Validate parameters
        limit = max(1, min(limit, 500))  # Cap at 500
        offset = max(0, offset)
        
        tracks = get_all_tracks(limit=limit, offset=offset, job_id=job_id)
        
        return jsonify({
            'tracks': tracks,
            'count': len(tracks),
            'limit': limit,
            'offset': offset,
            'job_id_filter': job_id,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get tracks: {str(e)}'}), 500


@bp.route('/tracks/<track_id>', methods=['GET'])
def get_track(track_id):
    """Get full details of a vehicle track including all cameras and timestamps."""
    try:
        track = get_vehicle_track(track_id)
        
        if track is None:
            return jsonify({'error': 'Track not found'}), 404
        
        return jsonify(track), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get track: {str(e)}'}), 500


@bp.route('/jobs/<job_id>/tracks', methods=['GET'])
def get_job_tracks(job_id):
    """Get all vehicle tracks that originated from a specific search job."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate parameters
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        
        tracks = get_all_tracks(limit=limit, offset=offset, job_id=job_id)
        
        return jsonify({
            'job_id': job_id,
            'tracks': tracks,
            'count': len(tracks),
            'limit': limit,
            'offset': offset,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get job tracks: {str(e)}'}), 500


@bp.route('/cameras/<camera_id>/tracks', methods=['GET'])
def get_camera_tracks(camera_id):
    """Get all vehicle tracks that include detections from a specific camera."""
    try:
        limit = request.args.get('limit', 50, type=int)
        limit = max(1, min(limit, 200))  # Cap at 200
        
        tracks = get_tracks_for_camera(camera_id, limit=limit)
        
        return jsonify({
            'camera_id': camera_id,
            'tracks': tracks,
            'count': len(tracks),
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to get camera tracks: {str(e)}'}), 500


@bp.route('/config', methods=['GET'])
def get_tracking_config():
    """Get current cross-camera tracking configuration."""
    return jsonify({
        'config': TRACKING_CONFIG,
    }), 200


@bp.route('/config', methods=['PUT'])
def update_tracking_config():
    """Update cross-camera tracking configuration."""
    try:
        data = request.get_json(force=True, silent=True)
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
                TRACKING_CONFIG[key] = value
        
        return jsonify({
            'message': 'Configuration updated',
            'config': TRACKING_CONFIG,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to update config: {str(e)}'}), 500
