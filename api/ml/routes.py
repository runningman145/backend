"""
ML model endpoints.
Provides status checks and video processing via the model pipeline.
"""
import base64
import cv2
import numpy as np
import uuid
import os
from io import BytesIO
from flask import Blueprint, jsonify, request, current_app
from .loader import get_models, MODEL_CONFIG
from .inference import extract_embedding, process_video_data
from ..db import get_db
from ..jobs.models import create_job, get_job_status, update_job_status

bp = Blueprint('model', __name__, url_prefix='/model')


def _load_video_from_request():
    """Load video from base64 data or file upload."""
    # Try file upload first
    if 'video' in request.files:
        video_file = request.files['video']
        return video_file.read()
    
    # Try base64 data
    if request.json and 'video_data' in request.json:
        video_b64 = request.json.get('video_data')
        if video_b64:
            try:
                # Handle data URL format if present
                if ',' in video_b64 and video_b64.startswith('data:'):
                    video_b64 = video_b64.split(',', 1)[1]
                return base64.b64decode(video_b64)
            except Exception as e:
                current_app.logger.error(f"Error decoding base64 video: {str(e)}")
                return None
    
    return None


def _load_image_from_request(field_name):
    """Load image from file upload or base64 data."""
    # Try file upload
    if field_name in request.files:
        image_file = request.files[field_name]
        if image_file.filename == '':
            return None
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            current_app.logger.error(f"Failed to decode image from file: {field_name}")
        return img
    
    # Try base64 data
    if request.json and field_name in request.json:
        b64_data = request.json.get(field_name)
        if b64_data:
            try:
                # Handle data URL format if present
                if ',' in b64_data and b64_data.startswith('data:'):
                    b64_data = b64_data.split(',', 1)[1]
                image_bytes = base64.b64decode(b64_data)
                image_array = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is None:
                    current_app.logger.error(f"Failed to decode image from base64: {field_name}")
                return img
            except Exception as e:
                current_app.logger.error(f"Error decoding base64 image: {str(e)}")
                return None
    
    return None


@bp.route('/status', methods=['GET'])
def model_status():
    """Check if models are loaded and ready."""
    try:
        models = get_models()
        device = models.get('device')
        device_str = str(device) if device else 'unknown'
        
        return jsonify({
            'status': 'ready' if models.get('yolo') is not None else 'not_loaded',
            'device': device_str,
            'reid_loaded': models.get('reid') is not None,
            'yolo_loaded': models.get('yolo') is not None,
        })
    except Exception as e:
        current_app.logger.error(f"Error checking model status: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@bp.route('/process-video', methods=['POST'])
def process_video():
    """
    Queue a video for background processing.
    
    Request body (JSON or multipart/form-data):
    {
        'video_data': '<base64-encoded video>',  // OR upload as multipart file 'video'
        'query_image': '<base64-encoded query image>',  // OR upload as multipart file 'query_image'
        'camera_id': '<camera ID>',
        'detection_id': '<optional detection_id to associate results>',
        'threshold': <optional similarity threshold, default 40>,
        'frame_skip': <optional frames to skip, default 15>
    }
    
    Returns:
    {
        'job_id': '<UUID of the queued job>',
        'status': 'queued',
        'message': 'Video queued for processing'
    }
    """
    db = None
    
    try:
        # Validate camera_id
        camera_id = None
        if request.json:
            camera_id = request.json.get('camera_id')
        elif request.form:
            camera_id = request.form.get('camera_id')
        
        if not camera_id:
            return jsonify({'error': 'camera_id is required'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (camera_id,)).fetchone()
        if camera is None:
            return jsonify({'error': f'Camera {camera_id} not found'}), 404
        
        # Load video
        video_data = _load_video_from_request()
        if not video_data:
            return jsonify({'error': 'Video data required (video_data or uploaded file)'}), 400
        
        # Load query image
        query_image = _load_image_from_request('query_image')
        if query_image is None:
            return jsonify({'error': 'Query image required'}), 400
        
        # Extract embedding from query image
        try:
            models = get_models()
            reid_model = models.get('reid')
            transform_func = models.get('transform')
            device = models.get('device')
            
            if reid_model is None or transform_func is None:
                return jsonify({'error': 'Models not loaded properly'}), 503
            
            # Convert BGR to RGB for embedding extraction
            query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            query_embedding = extract_embedding(query_image_rgb, reid_model, transform_func, device)
            
        except Exception as e:
            current_app.logger.error(f"Error extracting query embedding: {str(e)}")
            return jsonify({'error': f'Failed to extract query embedding: {str(e)}'}), 500
        
        # Save files to disk
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        video_filename = f"video_{uuid.uuid4()}.mp4"
        query_image_filename = f"query_{uuid.uuid4()}.jpg"
        
        video_path = os.path.join(upload_folder, video_filename)
        query_image_path = os.path.join(upload_folder, query_image_filename)
        
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        success = cv2.imwrite(query_image_path, query_image)
        if not success:
            return jsonify({'error': 'Failed to save query image'}), 500
        
        # Create detection record
        detection_id = None
        if request.json and request.json.get('detection_id'):
            detection_id = request.json.get('detection_id')
        elif request.form and request.form.get('detection_id'):
            detection_id = request.form.get('detection_id')
        else:
            cursor = db.execute(
                'INSERT INTO detections (camera_id, query_embedding) VALUES (?, ?)',
                (camera_id, query_embedding.tolist())
            )
            db.commit()
            detection_id = cursor.lastrowid
        
        # Get processing parameters
        threshold = MODEL_CONFIG.get('MATCH_THRESHOLD_PERCENT', 40)
        frame_skip = MODEL_CONFIG.get('FRAME_SKIP', 15)
        
        if request.json:
            threshold = request.json.get('threshold', threshold)
            frame_skip = request.json.get('frame_skip', frame_skip)
        elif request.form:
            if request.form.get('threshold'):
                threshold = float(request.form.get('threshold'))
            if request.form.get('frame_skip'):
                frame_skip = int(request.form.get('frame_skip'))
        
        # Queue job
        job_id = create_job(
            camera_id=camera_id,
            detection_id=detection_id,
            video_filename=video_filename,
            query_image_filename=query_image_filename,
            threshold=threshold,
            frame_skip=frame_skip,
            query_embedding=query_embedding.tolist()  # Store embedding for processing
        )
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video queued for processing',
            'detection_id': detection_id,
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error queuing video: {str(e)}")
        if db:
            db.rollback()
        return jsonify({'error': f'Failed to queue video: {str(e)}'}), 500


@bp.route('/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    """
    Get the status of a queued video processing job.
    
    Returns:
    {
        'id': '<job_id>',
        'status': 'pending|processing|completed|failed',
        'created_at': '<timestamp>',
        'started_at': '<timestamp or null>',
        'completed_at': '<timestamp or null>',
        'results': { ... } // only if completed
        'error': '<error message>' // only if failed
    }
    """
    try:
        job = get_job_status(job_id)
        if job is None:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': f'Failed to get job status: {str(e)}'}), 500


@bp.route('/extract-embedding', methods=['POST'])
def extract_embedding_endpoint():
    """
    Extract embedding from a single image without video processing.
    
    Request body (JSON or multipart/form-data):
    {
        'image': '<base64-encoded image>'  // OR upload as multipart file 'image'
    }
    
    Returns:
    {
        'embedding': [list of floats],
        'success': true
    }
    """
    try:
        # Load image
        image = _load_image_from_request('image')
        if image is None:
            return jsonify({'error': 'Image required'}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract embedding
        models = get_models()
        reid_model = models.get('reid')
        transform_func = models.get('transform')
        device = models.get('device')
        
        if reid_model is None or transform_func is None:
            return jsonify({'error': 'Models not loaded properly'}), 503
        
        embedding = extract_embedding(image_rgb, reid_model, transform_func, device)
        
        return jsonify({
            'success': True,
            'embedding': embedding.tolist(),
            'embedding_shape': embedding.shape[0]
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error extracting embedding: {str(e)}")
        return jsonify({'error': f'Failed to extract embedding: {str(e)}'}), 500


@bp.route('/compare-embeddings', methods=['POST'])
def compare_embeddings():
    """
    Compare two embeddings and return similarity score.
    
    Request body (JSON):
    {
        'embedding1': [list of floats],
        'embedding2': [list of floats]
    }
    
    Returns:
    {
        'similarity': float (0-100),
        'success': true
    }
    """
    try:
        from .reid import cosine_similarity
        
        data = request.json
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        embedding1 = np.array(data.get('embedding1'))
        embedding2 = np.array(data.get('embedding2'))
        
        if embedding1 is None or embedding2 is None:
            return jsonify({'error': 'Both embedding1 and embedding2 are required'}), 400
        
        similarity = cosine_similarity(embedding1, embedding2) * 100
        
        return jsonify({
            'success': True,
            'similarity': round(similarity, 2)
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error comparing embeddings: {str(e)}")
        return jsonify({'error': f'Failed to compare embeddings: {str(e)}'}), 500


@bp.route('/config', methods=['GET'])
def get_config():
    """
    Get current model configuration.
    
    Returns:
    {
        'frame_skip': int,
        'min_box_area': int,
        'match_threshold_percent': int,
        'yolo_imgsz': int,
        'vehicle_classes': list
    }
    """
    try:
        return jsonify({
            'frame_skip': MODEL_CONFIG.get('FRAME_SKIP'),
            'min_box_area': MODEL_CONFIG.get('MIN_BOX_AREA'),
            'match_threshold_percent': MODEL_CONFIG.get('MATCH_THRESHOLD_PERCENT'),
            'yolo_imgsz': MODEL_CONFIG.get('YOLO_IMGSZ'),
            'vehicle_classes': MODEL_CONFIG.get('VEHICLE_CLASSES'),
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting config: {str(e)}")
        return jsonify({'error': str(e)}), 500