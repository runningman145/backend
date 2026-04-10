"""
Video management and retrieval system.
Handles storing video metadata and retrieving video data for processing.
"""
import os
import uuid
from datetime import datetime, timedelta
from flask import current_app
from .db import get_db


def register_video(camera_id, filename, storage_path, captured_at, size_bytes=None, duration_seconds=None):
    """
    Register a new video from a camera in the system.
    
    Args:
        camera_id: UUID of the camera that captured the video
        filename: Original filename of the video
        storage_path: Path where video is stored (could be local path, S3 key, etc.)
        captured_at: Datetime when video was captured
        size_bytes: Optional file size in bytes
        duration_seconds: Optional video duration in seconds
    
    Returns:
        video_id (UUID)
    """
    try:
        video_id = str(uuid.uuid4())
        db = get_db()
        
        db.execute(
            '''INSERT INTO videos (id, camera_id, filename, storage_path, captured_at, size_bytes, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (video_id, camera_id, filename, storage_path, captured_at.isoformat(), size_bytes, duration_seconds)
        )
        db.commit()
        
        current_app.logger.info(f"Registered video {video_id} from camera {camera_id}")
        return video_id
    
    except Exception as e:
        current_app.logger.error(f"Error registering video: {str(e)}")
        raise


def get_video_metadata(video_id):
    """
    Get metadata for a specific video.
    """
    try:
        db = get_db()
        video = db.execute(
            'SELECT * FROM videos WHERE id = ?',
            (video_id,)
        ).fetchone()
        
        if video is None:
            return None
        
        return dict(video)
    except Exception as e:
        current_app.logger.error(f"Error getting video metadata: {str(e)}")
        return None


def get_camera_videos(camera_id, limit=100, offset=0, start_date=None, end_date=None):
    """
    Get all videos from a specific camera with optional date filtering.
    
    Args:
        camera_id: Camera UUID
        limit: Number of videos to return
        offset: Pagination offset
        start_date: Optional datetime to filter videos after this date
        end_date: Optional datetime to filter videos before this date
    
    Returns:
        List of video metadata dicts
    """
    try:
        db = get_db()
        
        query = 'SELECT * FROM videos WHERE camera_id = ?'
        params = [camera_id]
        
        if start_date:
            query += ' AND captured_at >= ?'
            params.append(start_date.isoformat())
        
        if end_date:
            query += ' AND captured_at <= ?'
            params.append(end_date.isoformat())
        
        query += ' ORDER BY captured_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        videos = db.execute(query, params).fetchall()
        
        return [dict(v) for v in videos]
    
    except Exception as e:
        current_app.logger.error(f"Error getting camera videos: {str(e)}")
        return []


def get_unprocessed_videos(camera_id=None, limit=50):
    """
    Get videos that haven't been processed yet.
    
    Args:
        camera_id: Optional filter by specific camera
        limit: Maximum number to return
    
    Returns:
        List of unprocessed video metadata
    """
    try:
        db = get_db()
        
        if camera_id:
            query = 'SELECT * FROM videos WHERE camera_id = ? AND processed = 0 ORDER BY captured_at ASC LIMIT ?'
            videos = db.execute(query, (camera_id, limit)).fetchall()
        else:
            query = 'SELECT * FROM videos WHERE processed = 0 ORDER BY captured_at ASC LIMIT ?'
            videos = db.execute(query, (limit,)).fetchall()
        
        return [dict(v) for v in videos]
    
    except Exception as e:
        current_app.logger.error(f"Error getting unprocessed videos: {str(e)}")
        return []


def mark_video_processed(video_id):
    """
    Mark a video as processed.
    """
    try:
        db = get_db()
        db.execute(
            'UPDATE videos SET processed = 1 WHERE id = ?',
            (video_id,)
        )
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error marking video as processed: {str(e)}")


def load_video_data(video_id):
    """
    Load video data from storage.
    
    Assumes video is stored at the path specified in storage_path.
    For more complex storage (S3, etc), extend this function.
    
    Returns:
        Video bytes or None if not found
    """
    try:
        video_meta = get_video_metadata(video_id)
        if not video_meta:
            return None
        
        storage_path = video_meta['storage_path']
        
        # If storage_path is relative, make it relative to instance folder
        if not os.path.isabs(storage_path):
            storage_path = os.path.join(current_app.instance_path, storage_path)
        
        if not os.path.exists(storage_path):
            current_app.logger.warning(f"Video file not found: {storage_path}")
            return None
        
        with open(storage_path, 'rb') as f:
            video_data = f.read()
        
        return video_data
    
    except Exception as e:
        current_app.logger.error(f"Error loading video data: {str(e)}")
        return None


def save_video_to_storage(video_data, filename):
    """
    Save video data to storage and return the storage path.
    
    Args:
        video_data: Bytes of video file
        filename: Original filename for reference
    
    Returns:
        Tuple of (storage_path, size_bytes)
    """
    try:
        upload_folder = os.path.join(current_app.instance_path, 'videos')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Generate unique filename
        video_id = str(uuid.uuid4())
        storage_filename = f"{video_id}_{filename}"
        storage_path = os.path.join(upload_folder, storage_filename)
        
        # Save file
        with open(storage_path, 'wb') as f:
            f.write(video_data)
        
        size_bytes = os.path.getsize(storage_path)
        
        # Return relative path for storage in DB
        relative_path = os.path.join('videos', storage_filename)
        
        return relative_path, size_bytes
    
    except Exception as e:
        current_app.logger.error(f"Error saving video: {str(e)}")
        raise


def get_videos_in_date_range(camera_id, start_date, end_date, limit=100):
    """
    Get all videos from a camera within a date range.
    Useful for batch processing time periods.
    
    Args:
        camera_id: Camera UUID
        start_date: Start of time range (datetime)
        end_date: End of time range (datetime)
        limit: Max videos to return
    
    Returns:
        List of video metadata sorted by capture time
    """
    return get_camera_videos(
        camera_id, 
        limit=limit, 
        start_date=start_date, 
        end_date=end_date
    )


def get_video_count_by_camera(camera_id, days=7):
    """
    Get count of videos from a camera in the last N days.
    """
    try:
        db = get_db()
        start_date = datetime.utcnow() - timedelta(days=days)
        
        count = db.execute(
            'SELECT COUNT(*) as count FROM videos WHERE camera_id = ? AND captured_at > ?',
            (camera_id, start_date.isoformat())
        ).fetchone()['count']
        
        return count
    
    except Exception as e:
        current_app.logger.error(f"Error counting videos: {str(e)}")
        return 0
