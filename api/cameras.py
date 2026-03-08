from flask import Blueprint, jsonify, request
from .db import get_db

bp = Blueprint('cameras', __name__, url_prefix='/cameras')


@bp.route('', methods=['POST'])
def create_camera():
    """Add a new camera."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    missing = [f for f in ('name', 'latitude', 'longitude') if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    db = get_db()
    cursor = db.execute(
        'INSERT INTO cameras (name, latitude, longitude) VALUES (?, ?, ?)',
        (data['name'], data['latitude'], data['longitude'])
    )
    db.commit()

    camera = db.execute(
        'SELECT id, name, latitude, longitude, created_at FROM cameras WHERE id = ?',
        (cursor.lastrowid,)
    ).fetchone()

    return jsonify(_serialize(camera)), 201


@bp.route('', methods=['GET'])
def list_cameras():
    """Return all cameras."""
    db = get_db()
    rows = db.execute(
        'SELECT id, name, latitude, longitude, created_at FROM cameras ORDER BY created_at DESC'
    ).fetchall()

    return jsonify([_serialize(row) for row in rows])


@bp.route('/<int:camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Return a single camera by ID."""
    camera = _get_or_404(camera_id)
    return jsonify(_serialize(camera))


@bp.route('/<int:camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Update name, latitude, and/or longitude of a camera."""
    _get_or_404(camera_id)

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    allowed = {'name', 'latitude', 'longitude'}
    fields = {k: v for k, v in data.items() if k in allowed}

    if not fields:
        return jsonify({'error': f'Provide at least one of: {", ".join(allowed)}'}), 400

    set_clause = ', '.join(f'{col} = ?' for col in fields)
    values = list(fields.values()) + [camera_id]

    db = get_db()
    db.execute(f'UPDATE cameras SET {set_clause} WHERE id = ?', values)
    db.commit()

    updated = db.execute(
        'SELECT id, name, latitude, longitude, created_at FROM cameras WHERE id = ?',
        (camera_id,)
    ).fetchone()

    return jsonify(_serialize(updated))


@bp.route('/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera and its related detections."""
    _get_or_404(camera_id)

    db = get_db()
    db.execute('DELETE FROM detections WHERE camera_id = ?', (camera_id,))
    db.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
    db.commit()

    return '', 204


# ---------- helpers ----------

def _get_or_404(camera_id):
    db = get_db()
    row = db.execute(
        'SELECT id, name, latitude, longitude, created_at FROM cameras WHERE id = ?',
        (camera_id,)
    ).fetchone()
    if row is None:
        from flask import abort
        abort(404, description=f'Camera {camera_id} not found')
    return row


def _serialize(row):
    return {
        'id': row['id'],
        'name': row['name'],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'created_at': row['created_at'].isoformat(),
    }
