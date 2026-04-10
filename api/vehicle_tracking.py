"""
Cross-camera vehicle tracking system.
Correlates vehicle detections across cameras to build tracks of vehicle movements.
"""
import json
import numpy as np
import uuid
from datetime import datetime, timedelta
from flask import current_app
from .db import get_db


# Configuration for cross-camera correlation
TRACKING_CONFIG = {
    'EMBEDDING_SIMILARITY_THRESHOLD': 0.65,  # Min cosine similarity to match
    'TIME_WINDOW_SECONDS': 120,  # Max time between detections to be same vehicle
    'MAX_DISTANCE_KM': 5,  # Max distance between cameras for correlation
    'SPATIAL_FILTER_ENABLED': True,  # Use camera location to help correlation
}


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
            similarity = np.dot(embedding, other_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(other_embedding) + 1e-6
            )
            
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


def get_vehicle_track(track_id):
    """
    Get all detections in a track with camera info.
    
    Returns:
        List of detections with camera names and coordinates
    """
    try:
        db = get_db()
        track = db.execute(
            '''SELECT vt.id, vt.first_seen, vt.last_seen, COUNT(td.id) as detection_count
               FROM vehicle_tracks vt
               LEFT JOIN track_detections td ON vt.id = td.track_id
               WHERE vt.id = ?
               GROUP BY vt.id''',
            (track_id,)
        ).fetchone()
        
        if not track:
            return None
        
        detections = db.execute(
            '''SELECT vd.id, vd.timestamp, vd.match_score, vd.box_x1, vd.box_y1, vd.box_x2, vd.box_y2, 
                      c.id as camera_id, c.name as camera_name, c.latitude, c.longitude, d.captured_at
               FROM track_detections td
               JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               JOIN cameras c ON vd.camera_id = c.id
               JOIN detections d ON vd.detection_id = d.id
               WHERE td.track_id = ?
               ORDER BY vd.timestamp ASC''',
            (track_id,)
        ).fetchall()
        
        return {
            'track_id': track['id'],
            'first_seen': track['first_seen'],
            'last_seen': track['last_seen'],
            'detection_count': track['detection_count'],
            'detections': [
                {
                    'detection_id': d['id'],
                    'timestamp': d['timestamp'],
                    'match_score': d['match_score'],
                    'bbox': {'x1': d['box_x1'], 'y1': d['box_y1'], 'x2': d['box_x2'], 'y2': d['box_y2']},
                    'camera': {
                        'id': d['camera_id'],
                        'name': d['camera_name'],
                        'latitude': d['latitude'],
                        'longitude': d['longitude'],
                    },
                    'captured_at': d['captured_at'],
                }
                for d in detections
            ]
        }
    
    except Exception as e:
        current_app.logger.error(f"Error getting vehicle track: {str(e)}")
        return None


def get_all_tracks(limit=100, offset=0):
    """
    Get all vehicle tracks with summary info.
    """
    try:
        db = get_db()
        tracks = db.execute(
            '''SELECT vt.id, vt.first_seen, vt.last_seen, 
                      COUNT(DISTINCT td.vehicle_detection_id) as detection_count,
                      COUNT(DISTINCT vd.camera_id) as camera_count
               FROM vehicle_tracks vt
               LEFT JOIN track_detections td ON vt.id = td.track_id
               LEFT JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               GROUP BY vt.id
               ORDER BY vt.last_seen DESC
               LIMIT ? OFFSET ?''',
            (limit, offset)
        ).fetchall()
        
        return [
            {
                'track_id': t['id'],
                'first_seen': t['first_seen'],
                'last_seen': t['last_seen'],
                'detection_count': t['detection_count'],
                'camera_count': t['camera_count'],
            }
            for t in tracks
        ]
    
    except Exception as e:
        current_app.logger.error(f"Error getting tracks: {str(e)}")
        return []


def get_tracks_for_camera(camera_id, limit=50):
    """
    Get all vehicle tracks that include detections from a specific camera.
    """
    try:
        db = get_db()
        tracks = db.execute(
            '''SELECT DISTINCT vt.id, vt.first_seen, vt.last_seen,
                      COUNT(DISTINCT td.vehicle_detection_id) as detection_count
               FROM vehicle_tracks vt
               JOIN track_detections td ON vt.id = td.track_id
               JOIN vehicle_detections vd ON td.vehicle_detection_id = vd.id
               WHERE vd.camera_id = ?
               GROUP BY vt.id
               ORDER BY vt.last_seen DESC
               LIMIT ?''',
            (camera_id, limit)
        ).fetchall()
        
        return [
            {
                'track_id': t['id'],
                'first_seen': t['first_seen'],
                'last_seen': t['last_seen'],
                'detection_count': t['detection_count'],
            }
            for t in tracks
        ]
    
    except Exception as e:
        current_app.logger.error(f"Error getting camera tracks: {str(e)}")
        return []


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
