"""
Upload endpoints.
Handles file uploads and downloads for detections.
"""
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app, send_file
from ..db import get_db
from ..jobs.models import create_job

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
    """Upload query image for cross-camera detection matching."""
    # Check if request has file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    camera_name = request.form.get('camera_name')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not camera_name:
        return jsonify({'error': 'camera_name is required'}), 400
    
    # Validate file type (images only for query)
    if not allowed_file(file.filename):
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return jsonify({'error': f'File type not allowed. Allowed: {allowed}'}), 400
    
    # Look up camera by name
    db = get_db()
    camera = db.execute(
        'SELECT id FROM cameras WHERE name = ?', (camera_name,)
    ).fetchone()
    
    if camera is None:
        return jsonify({'error': f"Camera '{camera_name}' not found"}), 404
    
    camera_id = camera['id']
    
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
    
    # Create a detection record for this query image
    try:
        detection = db.execute(
            'INSERT INTO detections (camera_id, captured_at) VALUES (?, ?)',
            (camera_id, datetime.now().isoformat())
        )
        db.commit()
        detection_id = detection.lastrowid
    except Exception as e:
        return jsonify({'error': f'Failed to create detection record: {str(e)}'}), 500
    
    # Create jobs for all videos from OTHER cameras
    job_ids = []
    try:
        # Get all videos from other cameras
        other_videos = db.execute(
            '''SELECT id, filename, camera_id FROM videos 
               WHERE camera_id != ? 
               ORDER BY camera_id, captured_at DESC''',
            (camera_id,)
        ).fetchall()
        
        # Create a job for each video to process against this query image
        for video in other_videos:
            job_id = create_job(
                camera_id=video['camera_id'],
                detection_id=detection_id,
                video_filename=video['filename'],
                query_image_filename=filename,
                threshold=40,
                frame_skip=15
            )
            job_ids.append({
                'job_id': job_id,
                'camera_id': video['camera_id'],
                'video_filename': video['filename']
            })
    except Exception as e:
        current_app.logger.error(f"Failed to create jobs: {str(e)}")
    
    return jsonify({
        'message': 'Query image uploaded and processing started',
        'filename': filename,
        'camera_name': camera_name,
        'camera_id': camera_id,
        'detection_id': detection_id,
        'upload_path': f'/uploads/{filename}',
        'jobs_created': len(job_ids),
        'jobs': job_ids
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
