"""
Vehicle detection storage.
Stores detected vehicles with their embeddings for tracking.
"""
import json
from flask import current_app
from ..db import get_db


def store_vehicle_detection(detection_id, camera_id, timestamp, box, embedding, match_score=None):
    """
    Store a detected vehicle with its embedding for tracking.
    
    Args:
        detection_id: ID of the parent detection record
        camera_id: Camera where vehicle was detected
        timestamp: Frame timestamp in seconds
        box: Dict with keys x1, y1, x2, y2 (bbox coordinates)
        embedding: numpy array of ReID embedding
        match_score: Optional similarity score to query image
    
    Returns:
        vehicle_detection_id (integer)
    """
    try:
        # Serialize embedding as JSON
        embedding_bytes = json.dumps(embedding.tolist()).encode('utf-8')
        
        box_area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
        
        db = get_db()
        cursor = db.execute(
            '''INSERT INTO vehicle_detections 
               (detection_id, camera_id, timestamp, box_x1, box_y1, box_x2, box_y2, box_area, embedding, match_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (detection_id, camera_id, timestamp,
             box['x1'], box['y1'], box['x2'], box['y2'],
             box_area, embedding_bytes, match_score)
        )
        db.commit()
        
        return cursor.lastrowid
    except Exception as e:
        current_app.logger.error(f"Error storing vehicle detection: {str(e)}")
        raise
