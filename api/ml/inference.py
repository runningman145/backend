"""
Inference helpers for video processing and embedding extraction.
Handles YOLO detection and ReID embedding generation.
"""
import base64
import cv2
import numpy as np
import tempfile
import torch
import os
from flask import current_app
from .loader import MODEL_CONFIG, get_models
from .reid import cosine_similarity


def _encode_frame_as_base64(bgr_crop: np.ndarray) -> str:
    """
    JPEG-encode a BGR numpy array and return a base64 data-URI string.

    Args:
        bgr_crop: BGR image array (vehicle crop)

    Returns:
        Base64-encoded JPEG as a data URI string, e.g.
        'data:image/jpeg;base64,/9j/4AAQ...'
    """
    success, buffer = cv2.imencode('.jpg', bgr_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ''
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def extract_embedding(image, model, transform_func, device):
    """
    Extract ReID embedding from an image.
    
    Args:
        image: numpy array of image (RGB format)
        model: ReID model instance
        transform_func: Torchvision transforms function
        device: torch device (cpu or cuda)
    
    Returns:
        Normalized embedding vector (numpy array)
    """
    image_tensor = transform_func(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    
    embedding = embedding.squeeze().cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def process_video_data(video_data, yolo_model, reid_model, transform_func, device,
                       query_embedding, threshold, frame_skip, detection_id=None, camera_id=None):
    """
    Process video bytes and return matches.
    
    If detection_id and camera_id provided, also stores vehicle detections for cross-camera tracking.
    
    Args:
        video_data: Raw video file bytes
        yolo_model: YOLO detection model
        reid_model: ReID embedding model
        transform_func: Image transformation function
        device: torch device
        query_embedding: Reference embedding to match against
        threshold: Similarity threshold (0-100 percent)
        frame_skip: Process every Nth frame
        detection_id: Optional detection ID to associate vehicles
        camera_id: Optional camera ID for cross-camera tracking
    
    Returns:
        List of match results (dicts with time, match_percent, box, optional track_id)
    
    Raises:
        ValueError: If video file cannot be opened
        Exception: If any error occurs during processing
    """
    # Import here to avoid circular imports
    from ..tracking.store import store_vehicle_detection
    from ..tracking.correlate import correlate_vehicle_detections
    
    # Save video to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name
    
    video = None
    stored_detections = []  # Store detection IDs for optional batch correlation
    
    try:
        video = cv2.VideoCapture(tmp_path)
        if not video.isOpened():
            raise ValueError("Could not open video file")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback if FPS cannot be read
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_id = 0
        processed_frames = 0
        
        # Log processing start
        current_app.logger.info(f"Starting video processing: {total_frames} total frames, skipping every {frame_skip} frames")
        
        while True:
            ret, frame = video.read()
            if not ret or frame is None:
                break
            
            frame_id += 1
            
            # Skip frames based on frame_skip parameter
            if frame_id % frame_skip != 0:
                continue
            
            processed_frames += 1
            
            try:
                # Get original frame dimensions
                h, w = frame.shape[:2]
                
                # Resize frame for YOLO processing
                small_frame = cv2.resize(frame, (MODEL_CONFIG['YOLO_IMGSZ'], MODEL_CONFIG['YOLO_IMGSZ']))
                
                # Run YOLO detection
                detections = yolo_model(small_frame, imgsz=MODEL_CONFIG['YOLO_IMGSZ'], verbose=False)[0]
                
                # Process each detection
                for box in detections.boxes:
                    cls = int(box.cls[0])
                    
                    # Filter for vehicle classes only
                    if cls not in MODEL_CONFIG['VEHICLE_CLASSES']:
                        continue
                    
                    # Get bounding box coordinates
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    # Scale coordinates back to original frame size
                    x1 = int(x1 * w / MODEL_CONFIG['YOLO_IMGSZ'])
                    x2 = int(x2 * w / MODEL_CONFIG['YOLO_IMGSZ'])
                    y1 = int(y1 * h / MODEL_CONFIG['YOLO_IMGSZ'])
                    y2 = int(y2 * h / MODEL_CONFIG['YOLO_IMGSZ'])
                    
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Filter out boxes that are too small
                    box_area = (x2 - x1) * (y2 - y1)
                    min_box_area = MODEL_CONFIG.get('MIN_BOX_AREA', 1000)
                    if box_area < min_box_area:
                        continue
                    
                    # Extract vehicle crop
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop is None or vehicle_crop.size == 0:
                        continue
                    
                    # Convert BGR to RGB for ReID model
                    vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                    
                    # Extract embedding
                    vehicle_embedding = extract_embedding(vehicle_crop_rgb, reid_model, transform_func, device)
                    
                    # Calculate similarity with query
                    similarity = cosine_similarity(query_embedding, vehicle_embedding) * 100
                    
                    # Check if match meets threshold
                    if similarity > threshold:
                        timestamp = frame_id / fps if fps > 0 else 0

                        # Encode the vehicle crop as base64 for downstream display
                        frame_image = _encode_frame_as_base64(vehicle_crop)

                        result = {
                            'time': round(timestamp, 2),
                            'match_percent': round(similarity, 2),
                            'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'frame_id': frame_id,
                            'frame_image': frame_image,  # base64 JPEG data-URI of the vehicle crop
                        }
                        
                        # Store vehicle detection for cross-camera tracking
                        if detection_id and camera_id:
                            try:
                                # Store detection in database
                                vehicle_det_id = store_vehicle_detection(
                                    detection_id=detection_id,
                                    camera_id=camera_id,
                                    timestamp=timestamp,
                                    box=result['box'],
                                    embedding=vehicle_embedding.tolist(),  # Convert numpy array to list for JSON storage
                                    match_score=similarity
                                )
                                
                                # Store for potential batch correlation
                                stored_detections.append({
                                    'detection_id': vehicle_det_id,
                                    'camera_id': camera_id,
                                    'timestamp': timestamp,
                                    'embedding': vehicle_embedding,
                                    'result_index': len(results)
                                })
                                
                                # Optionally correlate immediately (or do batch at end)
                                track_id = correlate_vehicle_detections(
                                    vehicle_det_id, camera_id, timestamp, vehicle_embedding
                                )
                                
                                if track_id:
                                    result['track_id'] = track_id
                                    
                            except Exception as e:
                                current_app.logger.warning(f"Error storing vehicle detection: {str(e)}")
                        
                        results.append(result)
                        
            except Exception as e:
                current_app.logger.error(f"Error processing frame {frame_id}: {str(e)}")
                continue  # Continue with next frame on error
        
        # Log completion
        current_app.logger.info(f"Video processing complete: processed {processed_frames} frames, found {len(results)} matches")
        
        # Batch correlate remaining detections if needed (optional)
        if detection_id and camera_id and stored_detections:
            try:
                # Optional: batch correlation for any missed matches
                for det in stored_detections:
                    if 'track_id' not in results[det['result_index']]:
                        track_id = correlate_vehicle_detections(
                            det['detection_id'], det['camera_id'], det['timestamp'], det['embedding']
                        )
                        if track_id:
                            results[det['result_index']]['track_id'] = track_id
            except Exception as e:
                current_app.logger.warning(f"Error in batch correlation: {str(e)}")
        
        return results
    
    except Exception as e:
        current_app.logger.error(f"Video processing failed: {str(e)}")
        raise
    
    finally:
        # Clean up resources
        if video:
            video.release()
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                current_app.logger.warning(f"Could not remove temp file {tmp_path}: {str(e)}")


def process_image_for_matching(image_bytes, yolo_model, reid_model, transform_func, device):
    """
    Process a single image to extract vehicle embeddings for matching.
    
    Args:
        image_bytes: Raw image file bytes
        yolo_model: YOLO detection model
        reid_model: ReID embedding model
        transform_func: Image transformation function
        device: torch device
    
    Returns:
        List of dictionaries containing vehicle detections with embeddings and bounding boxes
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    h, w = img.shape[:2]
    small_img = cv2.resize(img, (MODEL_CONFIG['YOLO_IMGSZ'], MODEL_CONFIG['YOLO_IMGSZ']))
    
    # Run detection
    detections = yolo_model(small_img, imgsz=MODEL_CONFIG['YOLO_IMGSZ'], verbose=False)[0]
    
    vehicles = []
    
    for box in detections.boxes:
        cls = int(box.cls[0])
        
        # Filter for vehicle classes
        if cls not in MODEL_CONFIG['VEHICLE_CLASSES']:
            continue
        
        # Get bounding box coordinates
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = coords
        
        # Scale back to original size
        x1 = int(x1 * w / MODEL_CONFIG['YOLO_IMGSZ'])
        x2 = int(x2 * w / MODEL_CONFIG['YOLO_IMGSZ'])
        y1 = int(y1 * h / MODEL_CONFIG['YOLO_IMGSZ'])
        y2 = int(y2 * h / MODEL_CONFIG['YOLO_IMGSZ'])
        
        # Ensure bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Extract crop
        vehicle_crop = img[y1:y2, x1:x2]
        if vehicle_crop is None or vehicle_crop.size == 0:
            continue
        
        # Convert to RGB
        vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
        
        # Extract embedding
        embedding = extract_embedding(vehicle_crop_rgb, reid_model, transform_func, device)
        
        vehicles.append({
            'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'embedding': embedding,
            'class_id': cls
        })
    
    return vehicles


def batch_extract_embeddings(images, reid_model, transform_func, device):
    """
    Extract embeddings for multiple images in batch for efficiency.
    
    Args:
        images: List of numpy arrays (RGB format)
        reid_model: ReID model instance
        transform_func: Torchvision transforms function
        device: torch device
    
    Returns:
        List of normalized embedding vectors (numpy arrays)
    """
    if not images:
        return []
    
    # Transform all images
    image_tensors = []
    for img in images:
        img_tensor = transform_func(img).unsqueeze(0)
        image_tensors.append(img_tensor)
    
    # Batch them together
    batch_tensor = torch.cat(image_tensors, dim=0).to(device)
    
    with torch.no_grad():
        embeddings = reid_model(batch_tensor)
    
    # Normalize and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    
    return list(embeddings_np)