"""
Upload endpoints.
Handles file uploads and downloads for detections.
"""
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request, current_app, send_file
from ..db import get_db
from ..jobs.models import create_job, create_batch_job

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    camera_id = request.form.get('camera_id')
    detection_id = request.form.get('detection_id')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400

    if not allowed_file(file.filename):
        allowed = ', '.join(ALLOWED_EXTENSIONS)
        return jsonify({'error': f'File type not allowed. Allowed: {allowed}'}), 400

    db = get_db()
    camera = db.execute(
        'SELECT id FROM cameras WHERE id = ?', (camera_id,)
    ).fetchone()

    if camera is None:
        return jsonify({'error': f"Camera '{camera_id}' not found"}), 404

    if detection_id:
        detection = db.execute(
            'SELECT id, camera_id FROM detections WHERE id = ?',
            (detection_id,)
        ).fetchone()

        if detection is None:
            return jsonify({'error': f'Detection {detection_id} not found'}), 404

        if detection['camera_id'] != camera_id:
            return jsonify({'error': f'Detection {detection_id} does not belong to camera {camera_id}'}), 404

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename

    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, filename)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    if not detection_id:
        try:
            detection = db.execute(
                'INSERT INTO detections (camera_id) VALUES (?)',
                (camera_id,)
            )
            db.commit()
            detection_id = detection.lastrowid
        except Exception as e:
            return jsonify({'error': f'Failed to create detection record: {str(e)}'}), 500

    job_ids = []
    try:
        other_videos = db.execute(
            '''SELECT id, filename, camera_id FROM videos
               WHERE camera_id != ?
               ORDER BY camera_id, captured_at DESC''',
            (camera_id,)
        ).fetchall()

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
        'camera_id': camera_id,
        'detection_id': str(detection_id),
        'upload_path': f'/uploads/{filename}',
        'jobs_created': len(job_ids),
        'jobs': job_ids
    }), 201


@bp.route('/batch', methods=['POST'])
def batch_upload_media():
    """
    Upload multiple query images (0-5) for cross-camera detection matching.

    Accepts either a single camera (backward-compat) or multiple cameras.
    When multiple cameras are provided the endpoint creates ONE batch job
    that processes all cameras instead of one job per camera.

    Request format (multipart/form-data):
    - files / file_0…file_4: 1-5 image files
    - camera_names: one or more camera name values (repeated field)
      OR camera_name: single camera name (backward-compat)
    - camera_time_ranges: JSON object keyed by camera name
      e.g. {"Main St": {"start_time": "08:00:00", "end_time": "10:00:00"}, ...}
    - job_date: YYYY-MM-DD
    - start_time / end_time: fallback time range when camera_time_ranges absent
    - threshold: float 0-1 (default 0.4)
    - frame_skip: int (default 15)
    """
    try:
        # ------------------------------------------------------------------
        # Resolve camera list
        # ------------------------------------------------------------------
        camera_names = request.form.getlist('camera_names')   # multi-camera (new)
        if not camera_names:
            # Backward-compat: single camera_name
            single = request.form.get('camera_name')
            if single:
                camera_names = [single]

        if not camera_names:
            return jsonify({'error': 'camera_name or camera_names is required'}), 400

        # ------------------------------------------------------------------
        # Common parameters
        # ------------------------------------------------------------------
        job_date   = request.form.get('job_date')
        # Default time range (used when no per-camera override is present)
        default_start = request.form.get('start_time', '00:00:00')
        default_end   = request.form.get('end_time',   '23:59:59')

        threshold  = float(request.form.get('threshold', 40))
        if threshold <= 1.0:
            threshold = threshold * 100        # 0-1 → 0-100 range
        frame_skip = int(request.form.get('frame_skip', 15))

        if not job_date or not default_start or not default_end:
            return jsonify({'error': 'job_date, start_time, and end_time are required'}), 400

        # Per-camera time-range overrides (keyed by camera name)
        camera_time_ranges: dict = {}
        time_ranges_raw = request.form.get('camera_time_ranges')
        if time_ranges_raw:
            try:
                camera_time_ranges = json.loads(time_ranges_raw)
            except (json.JSONDecodeError, TypeError):
                camera_time_ranges = {}

        # ------------------------------------------------------------------
        # File collection
        # ------------------------------------------------------------------
        files = request.files.getlist('files')
        if not files:
            files = [request.files.get(f'file_{i}') for i in range(5)
                     if f'file_{i}' in request.files]
        files = [f for f in files if f is not None]

        if len(files) == 0:
            return jsonify({
                'error': 'At least one file is required. '
                         'Send as "files" (multipart list) or "file_0", "file_1", etc.'
            }), 400

        if len(files) > 5:
            return jsonify({'error': 'Maximum 5 files allowed'}), 400

        # ------------------------------------------------------------------
        # Resolve cameras → DB rows
        # ------------------------------------------------------------------
        db = get_db()
        camera_entries = []   # [{camera_id, camera_name, start_time, end_time}]
        for name in camera_names:
            row = db.execute(
                'SELECT id FROM cameras WHERE name = ?', (name,)
            ).fetchone()
            if row is None:
                return jsonify({'error': f"Camera '{name}' not found"}), 404
            tr = camera_time_ranges.get(name, {})
            camera_entries.append({
                'camera_id':   row['id'],
                'camera_name': name,
                'start_time':  tr.get('start_time', default_start),
                'end_time':    tr.get('end_time',   default_end),
            })

        primary_camera_id = camera_entries[0]['camera_id']

        # ------------------------------------------------------------------
        # Save uploaded files
        # ------------------------------------------------------------------
        uploaded_filenames = []
        upload_folder = get_upload_folder()

        for file in files:
            if file.filename == '':
                return jsonify({'error': 'One or more files have no name'}), 400

            if not allowed_file(file.filename):
                allowed = ', '.join(ALLOWED_EXTENSIONS)
                return jsonify({'error': f'File type not allowed. Allowed: {allowed}'}), 400

            filename  = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename  = timestamp + filename
            filepath  = os.path.join(upload_folder, filename)

            try:
                file.save(filepath)
                uploaded_filenames.append(filename)
            except Exception as e:
                return jsonify({'error': f'Failed to save file {file.filename}: {str(e)}'}), 500

        # ------------------------------------------------------------------
        # Create a SINGLE batch job covering all cameras
        # ------------------------------------------------------------------
        try:
            job_id = create_batch_job(
                camera_id=primary_camera_id,
                query_image_filenames=uploaded_filenames,
                threshold=threshold,
                frame_skip=frame_skip,
                job_date=job_date,
                start_time=camera_entries[0]['start_time'],
                end_time=camera_entries[0]['end_time'],
                camera_ids=camera_entries,   # ← all cameras with their time ranges
            )
        except Exception as e:
            current_app.logger.error(f"Failed to create batch job: {str(e)}")
            return jsonify({'error': f'Failed to create batch job: {str(e)}'}), 500

        camera_names_list = [e['camera_name'] for e in camera_entries]

        return jsonify({
            'message': 'Batch job created — all cameras will be searched in a single job',
            'job_id':       job_id,
            'camera_names': camera_names_list,
            'camera_count': len(camera_entries),
            'job_date':     job_date,
            'file_count':   len(uploaded_filenames),
            'upload_paths': [f'/uploads/{f}' for f in uploaded_filenames],
            'filenames':    uploaded_filenames,
        }), 201

    except ValueError as e:
        return jsonify({'error': f'Invalid parameter format: {str(e)}'}), 400
    except Exception as e:
        current_app.logger.error(f"Error in batch upload: {str(e)}")
        return jsonify({'error': f'Failed to process batch upload: {str(e)}'}), 500


@bp.route('/<filename>', methods=['GET'])
def download_media(filename):
    """Download uploaded media file."""
    upload_folder = get_upload_folder()
    filepath = os.path.join(upload_folder, secure_filename(filename))

    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return jsonify({'error': f'File {filename} not found'}), 404

    if not os.path.abspath(filepath).startswith(os.path.abspath(upload_folder)):
        return jsonify({'error': 'Invalid file path'}), 403

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename
    )