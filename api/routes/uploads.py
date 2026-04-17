"""
Upload endpoints.
Handles file uploads and downloads for detections.
"""
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app, send_file
from ..db import get_db

bp = Blueprint('uploads', __name__, url_prefix='/uploads')

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


@bp.route('', methods=['POST'])
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


@bp.route('/<filename>', methods=['GET'])
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
    
    # Serve the file
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename
    )
