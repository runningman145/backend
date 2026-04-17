"""
Background job worker.
Processes video detection jobs from the queue.
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime
from flask import current_app
from ..db import get_db
from ..ml.loader import get_models
from ..ml.inference import extract_embedding, process_video_data
from .models import update_job_status


def process_job(job):
    """Process a single video detection job."""
    job_id = job['id']
    
    try:
        # Mark job as processing
        update_job_status(job_id, 'processing')
        
        # Load the video and query image from disk
        upload_folder = os.path.join(current_app.instance_path, 'uploads')
        
        video_path = os.path.join(upload_folder, job['video_filename'])
        query_image_path = os.path.join(upload_folder, job['query_image_filename'])
        
        if not os.path.exists(video_path) or not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Video or image file not found")
        
        # Read files
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        query_image = cv2.imread(query_image_path)
        
        if query_image is None:
            raise ValueError("Could not read query image")
        
        # Process video
        models = get_models()
        yolo_model = models['yolo']
        reid_model = models['reid']
        transform_func = models['transform']
        device = models['device']
        
        query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_embedding = extract_embedding(
            query_image_rgb, reid_model, transform_func, device
        )
        
        results = process_video_data(
            video_data, yolo_model, reid_model, transform_func, device,
            query_embedding, job['threshold'], job['frame_skip'],
            job['detection_id'], job['camera_id']  # Pass for vehicle tracking
        )
        
        # Store detection matches in database
        db = get_db()
        for result in results:
            db.execute(
                'INSERT INTO detection_matches (detection_id, similarity_score, timestamp) VALUES (?, ?, ?)',
                (job['detection_id'], result['match_percent'], result['time'])
            )
        db.commit()
        
        # Mark job as completed with results
        result_data = json.dumps({
            'matches': results,
            'total_matches': len(results),
        })
        update_job_status(job_id, 'completed', result_data=result_data)
        
        current_app.logger.info(f"Job {job_id} completed successfully with {len(results)} matches")
        
    except Exception as e:
        error_msg = str(e)
        current_app.logger.error(f"Job {job_id} failed: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)
