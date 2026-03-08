from flask import Blueprint, jsonify, request
from .db import get_db

bp = Blueprint('tracking', __name__, url_prefix='/tracking')


@bp.route('/detections', methods=['GET'])
def get_detections():
    """Return all detections with the capturing camera's coordinates."""
    db = get_db()
    rows = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        ORDER BY d.captured_at DESC
        '''
    ).fetchall()

    return jsonify([
        {
            'detection_id': row['id'],
            'captured_at': row['captured_at'].isoformat(),
            'camera': {
                'id': row['camera_id'],
                'name': row['camera_name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
            },
        }
        for row in rows
    ])


@bp.route('/detections', methods=['POST'])
def add_detection():
    """Record a new car detection submitted by the ML model."""
    data = request.get_json()

    if not data or 'camera_id' not in data:
        return jsonify({'error': 'camera_id is required'}), 400

    camera_id = data['camera_id']
    db = get_db()

    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()

    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    cursor = db.execute(
        'INSERT INTO detections (camera_id) VALUES (?)', (camera_id,)
    )
    db.commit()

    new_row = db.execute(
        '''
        SELECT
            d.id,
            d.captured_at,
            c.id   AS camera_id,
            c.name AS camera_name,
            c.latitude,
            c.longitude
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.id = ?
        ''',
        (cursor.lastrowid,)
    ).fetchone()

    return jsonify({
        'detection_id': new_row['id'],
        'captured_at': new_row['captured_at'].isoformat(),
        'camera': {
            'id': new_row['camera_id'],
            'name': new_row['camera_name'],
            'latitude': new_row['latitude'],
            'longitude': new_row['longitude'],
        },
    }), 201
