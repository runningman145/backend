"""
Camera management endpoints.
Handles CRUD operations for cameras.
"""
import uuid
from flask import Blueprint, jsonify, request
from ..db import get_db

bp = Blueprint('cameras', __name__, url_prefix='/cameras')


@bp.route('', methods=['POST'])
def create_camera():
    """Add a new camera."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    missing = [f for f in ('name', 'latitude', 'longitude') if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    camera_id = str(uuid.uuid4())
    status = data.get('status', 'offline')
    
    # Validate status value
    valid_statuses = {'online', 'offline', 'inactive'}
    if status not in valid_statuses:
        return jsonify({'error': f'Status must be one of: {", ".join(valid_statuses)}'}), 400
    
    db = get_db()
    db.execute(
        'INSERT INTO cameras (id, name, latitude, longitude, status) VALUES (?, ?, ?, ?, ?)',
        (camera_id, data['name'], data['latitude'], data['longitude'], status)
    )
    db.commit()

    camera = db.execute(
        'SELECT id, name, latitude, longitude, status, created_at FROM cameras WHERE id = ?',
        (camera_id,)
    ).fetchone()

    return jsonify(_serialize(camera)), 201


@bp.route('', methods=['GET'])
def list_cameras():
    """Return all cameras."""
    db = get_db()
    rows = db.execute(
        'SELECT id, name, latitude, longitude, status, created_at FROM cameras ORDER BY created_at DESC'
    ).fetchall()

    return jsonify([_serialize(row) for row in rows])


@bp.route('/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Return a single camera by ID."""
    camera = _get_or_404(camera_id)
    return jsonify(_serialize(camera))

@bp.route('/<camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Update name, latitude, longitude, and/or status of a camera."""
    _get_or_404(camera_id)

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    allowed = {'name', 'latitude', 'longitude', 'status'}
    fields = {k: v for k, v in data.items() if k in allowed}

    if not fields:
        return jsonify({'error': f'Provide at least one of: {", ".join(allowed)}'}), 400
    
    # Validate status if provided
    if 'status' in fields:
        valid_statuses = {'online', 'offline', 'inactive'}
        if fields['status'] not in valid_statuses:
            return jsonify({'error': f'Status must be one of: {", ".join(valid_statuses)}'}), 400

    set_clause = ', '.join(f'{col} = ?' for col in fields)
    values = list(fields.values()) + [camera_id]

    db = get_db()
    db.execute(f'UPDATE cameras SET {set_clause} WHERE id = ?', values)
    db.commit()

    updated = db.execute(
        'SELECT id, name, latitude, longitude, status, created_at FROM cameras WHERE id = ?',
        (camera_id,)
    ).fetchone()

    return jsonify(_serialize(updated))


@bp.route('/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera and its related detections."""
    _get_or_404(camera_id)

    db = get_db()
    db.execute('DELETE FROM detections WHERE camera_id = ?', (camera_id,))
    db.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    db.commit()

    return '', 204

@bp.route('/<camera_id>/videos/<video_id>', methods=['DELETE'])
def delete_camera_video(camera_id, video_id):
    """Delete a particular video from a camera."""
    _get_or_404(camera_id)

    db = get_db()
    db.execute('DELETE FROM videos WHERE id = ? AND camera_id = ?', (video_id, camera_id))
    db.commit()

    return '', 204

@bp.route('/<camera_id>/videos', methods=['GET'])
def list_camera_videos(camera_id):
    """List all videos from a specific camera."""
    _get_or_404(camera_id)
    
    db = get_db()
    limit = min(request.args.get('limit', 100, type=int), 1000)
    offset = request.args.get('offset', 0, type=int)
    
    videos = db.execute(
        '''SELECT id, filename, storage_path, size_bytes, duration_seconds, 
                  captured_at, created_at, processed 
           FROM videos 
           WHERE camera_id = ? 
           ORDER BY captured_at DESC 
           LIMIT ? OFFSET ?''',
        (camera_id, limit, offset)
    ).fetchall()
    
    total = db.execute(
        'SELECT COUNT(*) as count FROM videos WHERE camera_id = ?',
        (camera_id,)
    ).fetchone()['count']
    
    return jsonify({
        'camera_id': camera_id,
        'videos': [dict(v) for v in videos],
        'count': len(videos),
        'total': total,
        'limit': limit,
        'offset': offset
    }), 200


# ---------- helpers ----------

def _get_or_404(camera_id):
    """Get camera by ID or abort with 404."""
    db = get_db()
    row = db.execute(
        'SELECT id, name, latitude, longitude, status, created_at FROM cameras WHERE id = ?',
        (camera_id,)
    ).fetchone()
    if row is None:
        from flask import abort
        abort(404, description=f'Camera {camera_id} not found')
    return row


def _serialize(row):
    """Serialize camera database row to JSON-compatible dict."""
    return {
        'id': row['id'],
        'name': row['name'],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'status': row['status'],
        'created_at': row['created_at'].isoformat(),
    }
