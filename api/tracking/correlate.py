"""
Cross-camera vehicle correlation.
Correlates vehicle detections across cameras to build tracks.
"""
import json
import numpy as np
import uuid
from datetime import datetime, timezone
from flask import current_app
from ..db import get_db
from ..ml.reid import cosine_similarity


# Configuration for cross-camera correlation
TRACKING_CONFIG = {
    'EMBEDDING_SIMILARITY_THRESHOLD': 0.65,  # Min cosine similarity to match
    'TIME_WINDOW_SECONDS': 7200,  # Max time between detections to be same vehicle (2 hours)
    'MAX_DISTANCE_KM': 5,  # Max distance between cameras for correlation
    'SPATIAL_FILTER_ENABLED': True,  # Use camera location to help correlation
}


def correlate_vehicle_detections(new_vehicle_id, camera_id, timestamp, embedding, job_id=None, query_embedding=None):
    """
    Find similar vehicles across cameras and group them into tracks.
    
    Args:
        new_vehicle_id: ID of newly detected vehicle
        camera_id: Camera where vehicle was detected
        timestamp: When vehicle was detected
        embedding: numpy array of ReID embedding
        job_id: Optional ID of the job that triggered this detection (for filtering)
        query_embedding: Optional numpy array of the original query image embedding
    
    Returns:
        track_id: ID of track (new or existing)
    """
    try:
        db = get_db()
        
        # Query recent vehicle detections from OTHER cameras within the same job/search
        time_window_start = timestamp - TRACKING_CONFIG['TIME_WINDOW_SECONDS']
        
        # Build the base query - filter by job_id if provided
        if job_id:
            recent_detections = db.execute(
                '''SELECT vd.id, vd.camera_id, vd.timestamp, vd.embedding, vd.query_embedding, 
                          vt.id as track_id, c1.latitude, c1.longitude, c2.latitude as cam2_lat, c2.longitude as cam2_lon
                   FROM vehicle_detections vd
                   LEFT JOIN track_detections td ON vd.id = td.vehicle_detection_id
                   LEFT JOIN vehicle_tracks vt ON td.track_id = vt.id
                   JOIN cameras c1 ON vd.camera_id = c1.id
                   JOIN cameras c2 ON c2.id = ?
                   WHERE vd.camera_id != ? 
                   AND vd.job_id = ?
                   AND vd.timestamp > ?
                   AND vd.timestamp < ?
                   ORDER BY vd.timestamp DESC
                   LIMIT 100''',
                (camera_id, camera_id, job_id, time_window_start, timestamp)
            ).fetchall()
        else:
            # Fallback for legacy calls without job_id (shouldn't happen in normal operation)
            recent_detections = db.execute(
                '''SELECT vd.id, vd.camera_id, vd.timestamp, vd.embedding, vd.query_embedding, 
                          vt.id as track_id, c1.latitude, c1.longitude, c2.latitude as cam2_lat, c2.longitude as cam2_lon
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
            # If query_embedding is available, compare against that; otherwise use vehicle-to-vehicle
            if query_embedding is not None:
                # Compare new detection against the original query image
                similarity = cosine_similarity(query_embedding, other_embedding)
            else:
                # Compare against the detected vehicle's embedding
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
            _create_track(track_id, best_match, new_vehicle_id, camera_id, timestamp, job_id)
        else:
            # No match - create new track with just this vehicle
            track_id = str(uuid.uuid4())
            _create_single_vehicle_track(track_id, new_vehicle_id, camera_id, timestamp, job_id)
        
        return track_id
    
    except Exception as e:
        current_app.logger.error(f"Error correlating vehicle detections: {str(e)}")
        # Return new track on error
        track_id = str(uuid.uuid4())
        _create_single_vehicle_track(track_id, new_vehicle_id, camera_id, timestamp, job_id)
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


def _create_track(track_id, matched_detection, new_vehicle_id, camera_id, timestamp, job_id=None):
    """Create a new track linking two detected vehicles."""
    try:
        db = get_db()
        
        # Create track
        db.execute(
            '''INSERT INTO vehicle_tracks (id, job_id, first_camera_id, last_camera_id)
               VALUES (?, ?, ?, ?)''',
            (track_id, job_id, matched_detection['camera_id'], camera_id)
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


def _create_single_vehicle_track(track_id, vehicle_detection_id, camera_id, timestamp, job_id=None):
    """Create a track with just one vehicle detection."""
    try:
        db = get_db()
        
        db.execute(
            '''INSERT INTO vehicle_tracks (id, job_id, first_camera_id, last_camera_id)
               VALUES (?, ?, ?, ?)''',
            (track_id, job_id, camera_id, camera_id)
        )
        
        db.execute(
            'INSERT INTO track_detections (track_id, vehicle_detection_id) VALUES (?, ?)',
            (track_id, vehicle_detection_id)
        )
        
        db.commit()
    except Exception as e:
        current_app.logger.error(f"Error creating single vehicle track: {str(e)}")


def post_correlate_batch_detections(job_id, db):
    """
    Build cross-camera tracks after all cameras in a batch job are processed.

    Inline correlation (called per-video during processing) cannot link Camera A
    detections to Camera B because Camera B hasn't been stored yet when Camera A
    runs.  This function runs once after the entire job loop finishes:

    1. Clears any single-camera tracks created inline (they're incomplete).
    2. Fetches all vehicle_detections for the job.
    3. Uses union-find with a symmetric |t_a - t_b| <= TIME_WINDOW check on the
       absolute UTC epoch timestamps now stored in vehicle_detections.timestamp.
    4. Creates new tracks — multi-camera tracks where detections match, single-
       camera tracks otherwise.
    """
    from flask import current_app

    rows = db.execute(
        '''SELECT vd.id, vd.camera_id, vd.timestamp, vd.embedding
           FROM vehicle_detections vd
           WHERE vd.job_id = ?
           ORDER BY vd.timestamp ASC''',
        (job_id,)
    ).fetchall()

    if not rows:
        return

    # Remove inline-created tracks so we start fresh.
    old_track_ids = [
        r['id'] for r in db.execute(
            'SELECT id FROM vehicle_tracks WHERE job_id = ?', (job_id,)
        ).fetchall()
    ]
    for tid in old_track_ids:
        db.execute('DELETE FROM track_detections WHERE track_id = ?', (tid,))
    db.execute('DELETE FROM vehicle_tracks WHERE job_id = ?', (job_id,))
    db.commit()

    # Pre-parse everything once.
    ids = [r['id'] for r in rows]
    cameras = [r['camera_id'] for r in rows]
    timestamps = [float(r['timestamp']) for r in rows]
    embeddings: list = []
    for r in rows:
        try:
            embeddings.append(np.array(json.loads(r['embedding'])))
        except Exception:
            embeddings.append(None)

    # --- Union-Find ---
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    threshold = TRACKING_CONFIG['EMBEDDING_SIMILARITY_THRESHOLD']
    time_window = TRACKING_CONFIG['TIME_WINDOW_SECONDS']
    n = len(rows)

    # Rows are sorted by timestamp, so break the inner loop once the time gap
    # exceeds the window — no further j can satisfy the condition.
    for i in range(n):
        emb_i = embeddings[i]
        if emb_i is None:
            continue
        for j in range(i + 1, n):
            if timestamps[j] - timestamps[i] > time_window:
                break
            if cameras[j] == cameras[i]:        # same camera — skip
                continue
            emb_j = embeddings[j]
            if emb_j is None:
                continue
            if cosine_similarity(emb_i, emb_j) >= threshold:
                union(i, j)

    # Group indices by their union-find root.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Write tracks to DB.
    n_multi = 0
    for indices in groups.values():
        track_id = str(uuid.uuid4())
        cam_set = {cameras[i] for i in indices}
        first_cam = cameras[indices[0]]
        last_cam = cameras[indices[-1]]

        group_ts = [timestamps[i] for i in indices if timestamps[i] > 1_000_000_000]
        if group_ts:
            first_seen = datetime.fromtimestamp(min(group_ts), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            last_seen  = datetime.fromtimestamp(max(group_ts), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            first_seen = last_seen = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        db.execute(
            'INSERT INTO vehicle_tracks (id, job_id, first_camera_id, last_camera_id, first_seen, last_seen) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (track_id, job_id, first_cam, last_cam, first_seen, last_seen)
        )
        for i in indices:
            db.execute(
                'INSERT OR IGNORE INTO track_detections (track_id, vehicle_detection_id) '
                'VALUES (?, ?)',
                (track_id, ids[i])
            )
        if len(cam_set) > 1:
            n_multi += 1

    db.commit()
    current_app.logger.info(
        f"Job {job_id}: post-correlation → {len(groups)} tracks "
        f"({n_multi} spanning multiple cameras)"
    )


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
