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
                       query_embedding, threshold, frame_skip, detection_id=None, camera_id=None,
                       video_path=None):
    """
    Process a video and return matches against the query embedding.

    Accepts either a *video_path* (preferred – avoids a RAM copy) or legacy
    *video_data* bytes (written to a temp file for backward-compatibility).

    Args:
        video_data:      Raw video bytes OR None when video_path is given.
        yolo_model:      YOLO detection model.
        reid_model:      ReID embedding model.
        transform_func:  Image transformation function.
        device:          torch device.
        query_embedding: Reference embedding to match against.
        threshold:       Similarity threshold (0-100 percent).
        frame_skip:      Process every Nth frame.
        detection_id:    Optional detection ID for cross-camera tracking.
        camera_id:       Optional camera ID for cross-camera tracking.
        video_path:      Direct path to the video file (preferred over video_data).

    Returns:
        List of match dicts {time, match_percent, box, frame_id, frame_image, ...}.
    """
    # Import here to avoid circular imports
    from ..tracking.store import store_vehicle_detection
    from ..tracking.correlate import correlate_vehicle_detections

    # ------------------------------------------------------------------ #
    # Resolve the path to the video file                                   #
    # ------------------------------------------------------------------ #
    tmp_path = None          # only set when we created a temp file
    cleanup_tmp = False

    if video_path and os.path.exists(video_path):
        # Preferred: use the file on disk directly – no RAM copy needed
        source_path = video_path
    else:
        # Legacy fallback: caller passed raw bytes; write to a temp file
        if not video_data:
            raise ValueError("Either video_path or video_data must be provided")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name
        source_path = tmp_path
        cleanup_tmp = True

    video = None
    # Accumulate detections for batch DB correlation at the end
    pending_correlations = []

    try:
        video = cv2.VideoCapture(source_path)
        if not video.isOpened():
            raise ValueError("Could not open video file")

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24  # Default fallback if FPS cannot be read

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_id = 0
        processed_frames = 0

        current_app.logger.info(
            f"Starting video processing: {total_frames} total frames, "
            f"skipping every {frame_skip} frames"
        )

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
                h, w = frame.shape[:2]

                # Resize frame for YOLO processing
                small_frame = cv2.resize(frame, (MODEL_CONFIG['YOLO_IMGSZ'], MODEL_CONFIG['YOLO_IMGSZ']))

                # Run YOLO detection
                detections = yolo_model(small_frame, imgsz=MODEL_CONFIG['YOLO_IMGSZ'], verbose=False)[0]

                # -------------------------------------------------------- #
                # Collect all valid vehicle crops for this frame            #
                # -------------------------------------------------------- #
                frame_crops = []   # (vehicle_crop_bgr, box_coords)

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

                    # Clamp to frame bounds
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < MODEL_CONFIG.get('MIN_BOX_AREA', 1000):
                        continue

                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop is None or vehicle_crop.size == 0:
                        continue

                    frame_crops.append((vehicle_crop, (x1, y1, x2, y2)))

                if not frame_crops:
                    continue

                # -------------------------------------------------------- #
                # Batch ReID inference for all crops in this frame          #
                # One forward pass instead of N separate calls              #
                # -------------------------------------------------------- #
                rgb_crops = [
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    for crop, _ in frame_crops
                ]
                vehicle_embeddings = batch_extract_embeddings(
                    rgb_crops, reid_model, transform_func, device
                )

                timestamp = frame_id / fps

                for (vehicle_crop, (x1, y1, x2, y2)), vehicle_embedding in zip(frame_crops, vehicle_embeddings):
                    similarity = cosine_similarity(query_embedding, vehicle_embedding) * 100

                    if similarity <= threshold:
                        continue

                    frame_image = _encode_frame_as_base64(vehicle_crop)

                    result = {
                        'time': round(timestamp, 2),
                        'match_percent': round(similarity, 2),
                        'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'frame_id': frame_id,
                        'frame_image': frame_image,
                    }

                    # Queue detection for batch DB correlation after the loop
                    if detection_id and camera_id:
                        pending_correlations.append({
                            'vehicle_embedding': vehicle_embedding,
                            'timestamp': timestamp,
                            'box': result['box'],
                            'match_score': similarity,
                            'result_index': len(results),
                        })

                    results.append(result)

            except Exception as e:
                current_app.logger.error(f"Error processing frame {frame_id}: {str(e)}")
                continue

        current_app.logger.info(
            f"Video processing complete: processed {processed_frames} frames, "
            f"found {len(results)} matches"
        )

        # ---------------------------------------------------------------- #
        # Batch DB correlation — done once after the main loop             #
        # ---------------------------------------------------------------- #
        if detection_id and camera_id and pending_correlations:
            try:
                for item in pending_correlations:
                    vehicle_det_id = store_vehicle_detection(
                        detection_id=detection_id,
                        camera_id=camera_id,
                        timestamp=item['timestamp'],
                        box=item['box'],
                        embedding=item['vehicle_embedding'].tolist(),
                        match_score=item['match_score'],
                    )
                    track_id = correlate_vehicle_detections(
                        vehicle_det_id, camera_id,
                        item['timestamp'], item['vehicle_embedding']
                    )
                    if track_id:
                        results[item['result_index']]['track_id'] = track_id
            except Exception as e:
                current_app.logger.warning(f"Error in batch correlation: {str(e)}")

        return results

    except Exception as e:
        current_app.logger.error(f"Video processing failed: {str(e)}")
        raise

    finally:
        if video:
            video.release()
        if cleanup_tmp and tmp_path and os.path.exists(tmp_path):
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