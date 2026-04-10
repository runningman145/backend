"""
Vehicle detection and re-identification (ReID) model integration.
Processes videos to find vehicle matches using YOLO detection and ReID embeddings.
"""
import base64
import cv2
import numpy as np
import torch
import os
from io import BytesIO
from flask import Blueprint, jsonify, request, current_app
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50
from werkzeug.utils import secure_filename
from .db import get_db
from .jobs import create_job, get_job_status

bp = Blueprint('model', __name__, url_prefix='/model')

# Global model instances (loaded once at startup)
_models_cache = {
    'yolo': None,
    'reid': None,
    'transform': None,
    'device': None,
}

# Model configuration
MODEL_CONFIG = {
    'FRAME_SKIP': 15,
    'MIN_BOX_AREA': 5000,
    'MATCH_THRESHOLD_PERCENT': 40,
    'YOLO_IMGSZ': 640,
    'REID_INPUT_SIZE': (256, 128),
    'VEHICLE_CLASSES': [2, 5, 7],  # car, bus, truck
}


def _get_device():
    """Get the appropriate device (GPU or CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_models():
    """Load YOLO and ReID models (called once)."""
    if _models_cache['yolo'] is not None:
        return  # Already loaded

    device = _get_device()
    _models_cache['device'] = device
    
    current_app.logger.info(f"Loading models on device: {device}")
    
    # Load YOLO model
    _models_cache['yolo'] = YOLO("yolov8n.pt")
    
    # Load ReID model
    reid_model = resnet50(pretrained=False)
    reid_model.fc = torch.nn.Identity()
    
    # Try to load pre-trained weight
    reid_model_path = current_app.config.get('REID_MODEL_PATH')
    if reid_model_path:
        try:
            reid_model.load_state_dict(torch.load(reid_model_path, map_location=device))
            current_app.logger.info(f"ReID model loaded from {reid_model_path}")
        except FileNotFoundError:
            current_app.logger.warning(f"ReID model not found at {reid_model_path}, using untrained model")
    else:
        current_app.logger.warning("REID_MODEL_PATH not configured, using untrained model")
    
    reid_model = reid_model.to(device)
    reid_model.eval()
    _models_cache['reid'] = reid_model
    
    # Set up transforms
    _models_cache['transform'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_CONFIG['REID_INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    current_app.logger.info("Models loaded successfully")


def extract_embedding(image, model, transform_func, device):
    """Extract ReID embedding from image."""
    image_tensor = transform_func(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    
    embedding = embedding.squeeze().cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def _get_models():
    """Get cached models, loading if necessary."""
    _load_models()
    return _models_cache


@bp.route('/status', methods=['GET'])
def model_status():
    """Check if models are loaded and ready."""
    return jsonify({
        'status': 'ready' if _models_cache['yolo'] is not None else 'not_loaded',
        'device': _models_cache['device'] or 'unknown',
    })


@bp.route('/process-video', methods=['POST'])
def process_video():
    """
    Queue a video for background processing.
    
    Request body (JSON):
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
    try:
        # Validate camera_id
        camera_id = request.json.get('camera_id') if request.json else None
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
        
        # Save files to disk
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        import uuid
        video_filename = f"video_{uuid.uuid4()}.mp4"
        query_image_filename = f"query_{uuid.uuid4()}.jpg"
        
        video_path = os.path.join(upload_folder, video_filename)
        query_image_path = os.path.join(upload_folder, query_image_filename)
        
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        cv2.imwrite(query_image_path, query_image)
        
        # Create detection record
        detection_id = None
        if request.json.get('detection_id'):
            detection_id = request.json.get('detection_id')
        else:
            cursor = db.execute(
                'INSERT INTO detections (camera_id) VALUES (?)',
                (camera_id,)
            )
            db.commit()
            detection_id = cursor.lastrowid
        
        # Get processing parameters
        threshold = request.json.get('threshold', MODEL_CONFIG['MATCH_THRESHOLD_PERCENT']) if request.json else MODEL_CONFIG['MATCH_THRESHOLD_PERCENT']
        frame_skip = request.json.get('frame_skip', MODEL_CONFIG['FRAME_SKIP']) if request.json else MODEL_CONFIG['FRAME_SKIP']
        
        # Queue job
        job_id = create_job(
            camera_id=camera_id,
            detection_id=detection_id,
            video_filename=video_filename,
            query_image_filename=query_image_filename,
            threshold=threshold,
            frame_skip=frame_skip
        )
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video queued for processing',
            'detection_id': detection_id,
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error queuing video: {str(e)}")
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


def _load_video_from_request():
    """Load video from base64 data or file upload."""
    # Try file upload first
    if 'video' in request.files:
        video_file = request.files['video']
        return video_file.read()
    
    # Try base64 data
    video_b64 = request.json.get('video_data')
    if video_b64:
        try:
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
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Try base64 data
    b64_data = request.json.get(field_name)
    if b64_data:
        try:
            image_bytes = base64.b64decode(b64_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            current_app.logger.error(f"Error decoding base64 image: {str(e)}")
    
    return None


def _process_video_data(video_data, yolo_model, reid_model, transform_func, device,
                       query_embedding, threshold, frame_skip, detection_id=None, camera_id=None):
    """
    Process video bytes and return matches.
    If detection_id and camera_id provided, also stores vehicle detections for cross-camera tracking.
    """
    from . import vehicle_tracking
    
    # Save video to temp location
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name
    
    try:
        video = cv2.VideoCapture(tmp_path)
        if not video.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400
        
        fps = video.get(cv2.CAP_PROP_FPS)
        results = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = video.read()
                if not ret or frame is None:
                    break
                
                frame_id += 1
                
                if frame_id % frame_skip != 0:
                    continue
                
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (MODEL_CONFIG['YOLO_IMGSZ'], MODEL_CONFIG['YOLO_IMGSZ']))
                
                detections = yolo_model(small_frame, imgsz=MODEL_CONFIG['YOLO_IMGSZ'], verbose=False)[0]
                
                for box in detections.boxes:
                    cls = int(box.cls[0])
                    if cls not in MODEL_CONFIG['VEHICLE_CLASSES']:
                        continue
                    
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    # Scale coordinates back to original frame size
                    x1 = int(x1 * w / MODEL_CONFIG['YOLO_IMGSZ'])
                    x2 = int(x2 * w / MODEL_CONFIG['YOLO_IMGSZ'])
                    y1 = int(y1 * h / MODEL_CONFIG['YOLO_IMGSZ'])
                    y2 = int(y2 * h / MODEL_CONFIG['YOLO_IMGSZ'])
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < MODEL_CONFIG['MIN_BOX_AREA']:
                        continue
                    
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop is None or vehicle_crop.size == 0:
                        continue
                    
                    vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                    vehicle_embedding = extract_embedding(vehicle_crop_rgb, reid_model, transform_func, device)
                    
                    similarity = np.dot(query_embedding, vehicle_embedding) * 100
                    
                    if similarity > threshold:
                        timestamp = frame_id / fps if fps > 0 else 0
                        
                        result = {
                            'time': round(timestamp, 2),
                            'match_percent': round(similarity, 2),
                            'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        }
                        
                        # Store vehicle detection for cross-camera tracking
                        if detection_id and camera_id:
                            try:
                                vehicle_det_id = vehicle_tracking.store_vehicle_detection(
                                    detection_id=detection_id,
                                    camera_id=camera_id,
                                    timestamp=timestamp,
                                    box=result['box'],
                                    embedding=vehicle_embedding,
                                    match_score=similarity
                                )
                                
                                # Correlate with vehicles from other cameras
                                track_id = vehicle_tracking.correlate_vehicle_detections(
                                    vehicle_det_id, camera_id, timestamp, vehicle_embedding
                                )
                                
                                result['track_id'] = track_id
                            except Exception as e:
                                current_app.logger.warning(f"Error storing vehicle detection: {str(e)}")
                        
                        results.append(result)
        finally:
            video.release()
        
        return results
    
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
