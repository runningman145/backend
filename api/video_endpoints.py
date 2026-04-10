"""
API endpoints for video management and batch processing.
"""
from flask import Blueprint, jsonify, request, current_app
from datetime import datetime, timedelta
from . import video_management
from . import model as model_module
from .jobs import create_job
from .db import get_db

bp = Blueprint('videos', __name__, url_prefix='/videos')


@bp.route('/register', methods=['POST'])
def register_video():
    """
    Register a new video in the system.
    
    Request body:
    {
        'camera_id': '<camera UUID>',
        'filename': 'video.mp4',
        'storage_path': 'path/to/video.mp4',  // where the video is stored
        'captured_at': '2026-04-10T10:30:00',
        'size_bytes': 1024000,  // optional
        'duration_seconds': 120.5  // optional
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400
        
        required = ['camera_id', 'filename', 'storage_path', 'captured_at']
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (data['camera_id'],)).fetchone()
        if camera is None:
            return jsonify({'error': f"Camera {data['camera_id']} not found"}), 404
        
        # Parse captured_at
        try:
            captured_at = datetime.fromisoformat(data['captured_at'])
        except ValueError:
            return jsonify({'error': 'Invalid captured_at format, use ISO format'}), 400
        
        video_id = video_management.register_video(
            camera_id=data['camera_id'],
            filename=data['filename'],
            storage_path=data['storage_path'],
            captured_at=captured_at,
            size_bytes=data.get('size_bytes'),
            duration_seconds=data.get('duration_seconds')
        )
        
        return jsonify({
            'video_id': video_id,
            'message': 'Video registered successfully'
        }), 201
    
    except Exception as e:
        current_app.logger.error(f"Error registering video: {str(e)}")
        return jsonify({'error': f'Failed to register video: {str(e)}'}), 500


@bp.route('/by-camera/<camera_id>', methods=['GET'])
def list_camera_videos(camera_id):
    """
    List all videos from a specific camera.
    
    Query parameters:
        limit: Number of videos (default 100, max 1000)
        offset: Pagination offset (default 0)
        start_date: ISO datetime filter (optional)
        end_date: ISO datetime filter (optional)
    """
    try:
        limit = min(request.args.get('limit', 100, type=int), 1000)
        offset = request.args.get('offset', 0, type=int)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid start_date format'}), 400
        
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
            except ValueError:
                return jsonify({'error': 'Invalid end_date format'}), 400
        
        videos = video_management.get_camera_videos(
            camera_id, 
            limit=limit, 
            offset=offset,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'camera_id': camera_id,
            'videos': videos,
            'count': len(videos),
            'limit': limit,
            'offset': offset
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing camera videos: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500


@bp.route('/<video_id>/metadata', methods=['GET'])
def get_video_metadata(video_id):
    """
    Get metadata for a specific video.
    """
    try:
        video = video_management.get_video_metadata(video_id)
        if video is None:
            return jsonify({'error': 'Video not found'}), 404
        
        return jsonify(video), 200
    
    except Exception as e:
        current_app.logger.error(f"Error getting video metadata: {str(e)}")
        return jsonify({'error': f'Failed to get metadata: {str(e)}'}), 500


@bp.route('/unprocessed/list', methods=['GET'])
def list_unprocessed_videos():
    """
    List videos that haven't been processed yet.
    
    Query parameters:
        camera_id: Optional filter by specific camera
        limit: Max videos to return (default 50)
    """
    try:
        camera_id = request.args.get('camera_id')
        limit = min(request.args.get('limit', 50, type=int), 500)
        
        videos = video_management.get_unprocessed_videos(camera_id=camera_id, limit=limit)
        
        return jsonify({
            'videos': videos,
            'count': len(videos),
            'filter_camera': camera_id
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error listing unprocessed videos: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500


@bp.route('/<video_id>/process', methods=['POST'])
def process_video(video_id):
    """
    Queue a video for processing with a query image.
    
    Request body:
    {
        'query_image': '<base64 encoded image OR file upload>',
        'threshold': 40,  // optional similarity threshold
        'frame_skip': 15  // optional frame skip
    }
    """
    try:
        video_meta = video_management.get_video_metadata(video_id)
        if video_meta is None:
            return jsonify({'error': 'Video not found'}), 404
        
        # Load video data
        video_data = video_management.load_video_data(video_id)
        if video_data is None:
            return jsonify({'error': 'Could not load video file'}), 500
        
        # Load query image from request
        import cv2
        import numpy as np
        import base64
        
        query_image = None
        
        # Try file upload first
        if 'query_image' in request.files:
            query_file = request.files['query_image']
            image_bytes = query_file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            query_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Try base64
        elif request.json and request.json.get('query_image'):
            try:
                image_bytes = base64.b64decode(request.json.get('query_image'))
                image_array = np.frombuffer(image_bytes, np.uint8)
                query_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                current_app.logger.error(f"Error decoding query image: {str(e)}")
        
        if query_image is None:
            return jsonify({'error': 'Query image required'}), 400
        
        # Save query image temporarily
        import os
        import uuid
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        query_image_filename = f"query_{uuid.uuid4()}.jpg"
        query_image_path = os.path.join(upload_folder, query_image_filename)
        cv2.imwrite(query_image_path, query_image)
        
        # Save video temporarily
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(upload_folder, video_filename)
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        # Get processing parameters
        threshold = request.args.get('threshold', 40, type=float) if request.args else 40
        frame_skip = request.args.get('frame_skip', 15, type=int) if request.args else 15
        
        # Create detection record
        from .db import get_db
        db = get_db()
        cursor = db.execute(
            'INSERT INTO detections (camera_id) VALUES (?)',
            (video_meta['camera_id'],)
        )
        db.commit()
        detection_id = cursor.lastrowid
        
        # Queue job
        job_id = create_job(
            camera_id=video_meta['camera_id'],
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
            'video_id': video_id,
            'detection_id': detection_id
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500


@bp.route('/batch/process', methods=['POST'])
def batch_process_videos():
    """
    Queue multiple videos from a camera for batch processing.
    
    Request body:
    {
        'camera_id': '<camera UUID>',
        'query_image': '<base64 OR file upload>',
        'start_date': '2026-04-10T00:00:00',  // optional
        'end_date': '2026-04-10T23:59:59',    // optional
        'limit': 20,  // max videos to process
        'threshold': 40,
        'frame_skip': 15
    }
    """
    try:
        data = request.json or {}
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({'error': 'camera_id required'}), 400
        
        # Verify camera exists
        db = get_db()
        camera = db.execute('SELECT id FROM cameras WHERE id = ?', (camera_id,)).fetchone()
        if camera is None:
            return jsonify({'error': f'Camera {camera_id} not found'}), 404
        
        # Get date range
        start_date = None
        end_date = None
        
        if data.get('start_date'):
            try:
                start_date = datetime.fromisoformat(data['start_date'])
            except ValueError:
                return jsonify({'error': 'Invalid start_date format'}), 400
        
        if data.get('end_date'):
            try:
                end_date = datetime.fromisoformat(data['end_date'])
            except ValueError:
                return jsonify({'error': 'Invalid end_date format'}), 400
        
        limit = data.get('limit', 20)
        
        # Get videos
        videos = video_management.get_videos_in_date_range(
            camera_id, start_date, end_date, limit=limit
        ) if start_date else video_management.get_unprocessed_videos(camera_id, limit)
        
        if not videos:
            return jsonify({'error': 'No videos found matching criteria'}), 404
        
        # Load query image
        import cv2
        import numpy as np
        import base64
        
        query_image = None
        
        if 'query_image' in request.files:
            query_file = request.files['query_image']
            image_bytes = query_file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            query_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif data.get('query_image'):
            try:
                image_bytes = base64.b64decode(data['query_image'])
                image_array = np.frombuffer(image_bytes, np.uint8)
                query_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                current_app.logger.error(f"Error decoding query image: {str(e)}")
        
        if query_image is None:
            return jsonify({'error': 'Query image required'}), 400
        
        # Save query image
        import os
        import uuid
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        query_image_filename = f"query_{uuid.uuid4()}.jpg"
        query_image_path = os.path.join(upload_folder, query_image_filename)
        cv2.imwrite(query_image_path, query_image)
        
        # Queue jobs for all videos
        threshold = data.get('threshold', 40)
        frame_skip = data.get('frame_skip', 15)
        
        job_ids = []
        for video_meta in videos:
            try:
                video_data = video_management.load_video_data(video_meta['id'])
                if not video_data:
                    continue
                
                # Save video
                video_filename = f"video_{uuid.uuid4()}.mp4"
                video_path = os.path.join(upload_folder, video_filename)
                with open(video_path, 'wb') as f:
                    f.write(video_data)
                
                # Create detection
                cursor = db.execute(
                    'INSERT INTO detections (camera_id) VALUES (?)',
                    (camera_id,)
                )
                db.commit()
                detection_id = cursor.lastrowid
                
                # Queue job
                job_id = create_job(
                    camera_id=camera_id,
                    detection_id=detection_id,
                    video_filename=video_filename,
                    query_image_filename=query_image_filename,
                    threshold=threshold,
                    frame_skip=frame_skip
                )
                job_ids.append(job_id)
            
            except Exception as e:
                current_app.logger.warning(f"Failed to queue video {video_meta['id']}: {str(e)}")
        
        return jsonify({
            'message': f'Queued {len(job_ids)} videos for processing',
            'camera_id': camera_id,
            'job_ids': job_ids,
            'total_queued': len(job_ids),
            'total_videos': len(videos)
        }), 202
    
    except Exception as e:
        current_app.logger.error(f"Error batch processing videos: {str(e)}")
        return jsonify({'error': f'Failed to process videos: {str(e)}'}), 500
