import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app
from .db import get_db

bp = Blueprint('tracking', __name__, url_prefix='/tracking')

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_upload_folder():
    """Get or create upload folder."""
    upload_folder = os.path.join(current_app.instance_path, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    return upload_folder


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


@bp.route('/upload', methods=['POST'])
def upload_media():
    """Upload picture or video file for a detection."""
    # Check if request has file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    camera_id = request.form.get('camera_id')
    detection_id = request.form.get('detection_id')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return jsonify({'error': f'File type not allowed. Allowed: {allowed}'}), 400
    
    # Verify camera exists
    db = get_db()
    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()
    
    if camera is None:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404
    
    # If detection_id provided, verify it exists
    if detection_id:
        detection = db.execute(
            'SELECT id FROM detections WHERE id = ? AND camera_id = ?',
            (detection_id, camera_id)
        ).fetchone()
        
        if detection is None:
            return jsonify({'error': f'Detection {detection_id} not found for camera {camera_id}'}), 404
    
    # Save file securely
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    
    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, filename)
    
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    return jsonify({
        'message': 'File uploaded successfully',
        'filename': filename,
        'camera_id': camera_id,
        'detection_id': detection_id,
        'upload_path': f'/uploads/{filename}'
    }), 201


@bp.route('/uploads/<filename>', methods=['GET'])
def download_media(filename):
    """Download uploaded media file."""
    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, secure_filename(filename))
    
    # Security: check file exists and is in upload folder
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return jsonify({'error': f'File {filename} not found'}), 404
    
    # Verify the file is actually in the upload folder
    if not os.path.abspath(filepath).startswith(os.path.abspath(upload_folder)):
        return jsonify({'error': 'Invalid file path'}), 403
    
    return jsonify({
        'filename': filename,
        'path': filepath,
        'size': os.path.getsize(filepath)
    })
