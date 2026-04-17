"""
Cross-camera vehicle correlation.
Correlates vehicle detections across cameras to build tracks.
"""
import json
import numpy as np
import uuid
from flask import current_app
from ..db import get_db
from ..ml.reid import cosine_similarity


# Configuration for cross-camera correlation
TRACKING_CONFIG = {
    'EMBEDDING_SIMILARITY_THRESHOLD': 0.65,  # Min cosine similarity to match
    'TIME_WINDOW_SECONDS': 120,  # Max time between detections to be same vehicle
    'MAX_DISTANCE_KM': 5,  # Max distance between cameras for correlation
    'SPATIAL_FILTER_ENABLED': True,  # Use camera location to help correlation
}


def correlate_vehicle_detections(new_vehicle_id, camera_id, timestamp, embedding):
    """
    Find similar vehicles across cameras and group them into tracks.
    
    Args:
        new_vehicle_id: ID of newly detected vehicle
        camera_id: Camera where vehicle was detected
        timestamp: When vehicle was detected
        embedding: numpy array of ReID embedding
    
    Returns:
        track_id: ID of track (new or existing)
    """
    try:
        db = get_db()
        
        # Query recent vehicle detections from other cameras
        time_window_start = timestamp - TRACKING_CONFIG['TIME_WINDOW_SECONDS']
        
        recent_detections = db.execute(
            '''SELECT vd.id, vd.camera_id, vd.timestamp, vd.embedding, vt.id as track_id, c1.latitude, c1.longitude, c2.latitude as cam2_lat, c2.longitude as cam2_lon
               FROM vehicle_detections vd
               LEFT JOIN track_detections td ON vd.id = td.vehicle_detection_id
               LEFT JOIN vehicle_tracks vt ON td.track_id = vt.id
               JOIN cameras c1 ON vd.camera_id = c1.id
               JOIN cameras c2 ON c2.id = ?
               WHERE vd.camera_id != ? 
               AND vd.timestamp > ?
               AND vd.timestamp < ?
               ORDER BY vd.timestamp DESC
               LIMIT 100''',
            (camera_id, camera_id, time_window_start, timestamp)
        ).fetchall()
        
        best_match = None
        best_similarity = TRACKING_CONFIG['EMBEDDING_SIMILARITY_THRESHOLD']
        
        for detection in recent_detections:
            # Deserialize embedding
            other_embedding = np.array(json.loads(detection['embedding']))
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding, other_embedding)
            
            # Check spatial proximity if enabled
            if TRACKING_CONFIG['SPATIAL_FILTER_ENABLED']:
                distance_km = _calculate_distance(
                    detection['latitude'], detection['longitude'],
                    detection['cam2_lat'], detection['cam2_lon']
                )
                
                if distance_km > TRACKING_CONFIG['MAX_DISTANCE_KM']:
                    continue  # Cameras too far apart
            
            # Update best match
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = detection
        
        # Assign to track
        track_id = None
        if best_match and best_match['track_id']:
            # Add to existing track
            track_id = best_match['track_id']
            _add_vehicle_to_track(new_vehicle_id, track_id)
        elif best_match:
            # Create new track from matched pair
            track_id = str(uuid.uuid4())
            _create_track(track_id, best_match, new_vehicle_id, camera_id, timestamp)
        else:
            # No match - create new track with just this vehicle
            track_id = str(uuid.uuid4())
            _create_single_vehicle_track(track_id, new_vehicle_id, camera_id, timestamp)
        
        return track_id
    
    except Exception as e:
        current_app.logger.error(f"Error correlating vehicle detections: {str(e)}")
        # Return new track on error
        track_id = str(uuid.uuid4())
        _create_single_vehicle_track(track_id, new_vehicle_id, camera_id, timestamp)
        return track_id


def _add_vehicle_to_track(vehicle_detection_id, track_id):
    """Add vehicle detection to an existing track."""
    try:
        db = get_db()
        db.execute(
            'INSERT OR IGNORE INTO track_detections (track_id, vehicle_detection_id) VALUES (?, ?)',
            (track_id, vehicle_detection_id)
        )
        
        # Update track's last_seen timestamp
        db.execute(
            '''UPDATE vehicle_tracks SET last_seen = CURRENT_TIMESTAMP 
               WHERE id = ?''',
            (track_id,)
        )
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error adding vehicle to track: {str(e)}")


def _create_track(track_id, matched_detection, new_vehicle_id, camera_id, timestamp):
    """Create a new track linking two detected vehicles."""
    try:
        db = get_db()
        
        # Create track
        db.execute(
            '''INSERT INTO vehicle_tracks (id, first_camera_id, last_camera_id)
               VALUES (?, ?, ?)''',
            (track_id, matched_detection['camera_id'], camera_id)
        )
        
        # Add both vehicles to track
        db.execute(
            'INSERT INTO track_detections (track_id, vehicle_detection_id) VALUES (?, ?)',
            (track_id, matched_detection['id'])
        )
        db.execute(
            'INSERT INTO track_detections (track_id, vehicle_detection_id) VALUES (?, ?)',
            (track_id, new_vehicle_id)
        )
        
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error creating track: {str(e)}")


def _create_single_vehicle_track(track_id, vehicle_detection_id, camera_id, timestamp):
    """Create a track with just one vehicle detection."""
    try:
        db = get_db()
        
        db.execute(
            '''INSERT INTO vehicle_tracks (id, first_camera_id, last_camera_id)
               VALUES (?, ?, ?)''',
            (track_id, camera_id, camera_id)
        )
        
        db.execute(
            'INSERT INTO track_detections (track_id, vehicle_detection_id) VALUES (?, ?)',
            (track_id, vehicle_detection_id)
        )
        
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error creating single vehicle track: {str(e)}")


def _calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance between two GPS coordinates in km."""
    try:
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        lon_delta = lon2 - lon1
        lat_delta = lat2 - lat1
        
        a = sin(lat_delta / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lon_delta / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        
        return km
    except Exception as e:
        current_app.logger.error(f"Error calculating distance: {str(e)}")
        return float('inf')  # Return large distance on error
