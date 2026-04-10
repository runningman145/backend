"""
Background job queue system for video processing.
Handles queueing, processing, and status tracking of video detection jobs.
"""
import json
import threading
import uuid
from datetime import datetime
from flask import current_app
from .db import get_db


class JobQueue:
    """Thread-based job queue for processing video detection tasks."""
    
    def __init__(self, num_workers=2):
        """Initialize job queue with specified number of worker threads."""
        self.num_workers = num_workers
        self.worker_threads = []
        self.running = False
    
    def start(self, app):
        """Start worker threads."""
        self.running = True
        self.app = app
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"JobWorker-{i}"
            )
            worker.start()
            self.worker_threads.append(worker)
        
        current_app.logger.info(f"Started {self.num_workers} job workers")
    
    def stop(self):
        """Stop worker threads gracefully."""
        self.running = False
        for worker in self.worker_threads:
            worker.join(timeout=5)
        current_app.logger.info("Job workers stopped")
    
    def _worker_loop(self):
        """Main loop for worker thread - processes jobs from queue."""
        with self.app.app_context():
            while self.running:
                job = self._get_next_pending_job()
                if job:
                    self._process_job(job)
                else:
                    # No pending jobs, sleep briefly to avoid busy waiting
                    threading.Event().wait(0.5)
    
    def _get_next_pending_job(self):
        """Get the next pending job from the queue."""
        try:
            db = get_db()
            job = db.execute(
                'SELECT id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip '
                'FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1',
                ('pending',)
            ).fetchone()
            return job
        except Exception as e:
            current_app.logger.error(f"Error fetching pending job: {str(e)}")
            return None
    
    def _process_job(self, job):
        """Process a single job."""
        job_id = job['id']
        
        try:
            # Mark job as processing
            self._update_job_status(job_id, 'processing')
            
            # Import here to avoid circular imports
            from . import model
            from . import vehicle_tracking
            
            # Load the video and query image from disk
            import os
            upload_folder = os.path.join(current_app.instance_path, 'uploads')
            
            video_path = os.path.join(upload_folder, job['video_filename'])
            query_image_path = os.path.join(upload_folder, job['query_image_filename'])
            
            if not os.path.exists(video_path) or not os.path.exists(query_image_path):
                raise FileNotFoundError(f"Video or image file not found")
            
            # Read files
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            import cv2
            import numpy as np
            query_image = cv2.imread(query_image_path)
            
            if query_image is None:
                raise ValueError("Could not read query image")
            
            # Process video
            models = model._get_models()
            yolo_model = models['yolo']
            reid_model = models['reid']
            transform_func = models['transform']
            device = models['device']
            
            query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            query_embedding = model.extract_embedding(
                query_image_rgb, reid_model, transform_func, device
            )
            
            results = model._process_video_data(
                video_data, yolo_model, reid_model, transform_func, device,
                query_embedding, job['threshold'], job['frame_skip'],
                job['detection_id'], job['camera_id']  # Pass for vehicle tracking
            )
            
            if isinstance(results, tuple):  # Error case
                raise Exception(f"Video processing failed: {results[0].json['error']}")
            
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
            self._update_job_status(job_id, 'completed', result_data=result_data)
            
            current_app.logger.info(f"Job {job_id} completed successfully with {len(results)} matches")
            
        except Exception as e:
            error_msg = str(e)
            current_app.logger.error(f"Job {job_id} failed: {error_msg}")
            self._update_job_status(job_id, 'failed', error_message=error_msg)
    
    def _update_job_status(self, job_id, status, result_data=None, error_message=None):
        """Update job status in database."""
        try:
            db = get_db()
            
            update_fields = ['status = ?']
            values = [status]
            
            if status == 'processing':
                update_fields.append('started_at = ?')
                values.append(datetime.utcnow().isoformat())
            elif status in ('completed', 'failed'):
                update_fields.append('completed_at = ?')
                values.append(datetime.utcnow().isoformat())
            
            if result_data is not None:
                update_fields.append('result_data = ?')
                values.append(result_data)
            
            if error_message is not None:
                update_fields.append('error_message = ?')
                values.append(error_message)
            
            values.append(job_id)
            
            set_clause = ', '.join(update_fields)
            db.execute(f'UPDATE jobs SET {set_clause} WHERE id = ?', values)
            db.commit()
        except Exception as e:
            current_app.logger.error(f"Error updating job status: {str(e)}")


# Global job queue instance
_job_queue = None


def get_job_queue():
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(num_workers=2)
    return _job_queue


def create_job(camera_id, detection_id, video_filename, query_image_filename, threshold=40, frame_skip=15):
    """Create a new job in the queue."""
    job_id = str(uuid.uuid4())
    db = get_db()
    
    db.execute(
        'INSERT INTO jobs (id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip, status) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (job_id, camera_id, detection_id, video_filename, query_image_filename, threshold, frame_skip, 'pending')
    )
    db.commit()
    
    return job_id


def get_job_status(job_id):
    """Get job status and details."""
    db = get_db()
    job = db.execute(
        'SELECT id, status, result_data, error_message, created_at, started_at, completed_at FROM jobs WHERE id = ?',
        (job_id,)
    ).fetchone()
    
    if job is None:
        return None
    
    result = {
        'id': job['id'],
        'status': job['status'],
        'created_at': job['created_at'],
        'started_at': job['started_at'],
        'completed_at': job['completed_at'],
    }
    
    if job['result_data']:
        result['results'] = json.loads(job['result_data'])
    
    if job['error_message']:
        result['error'] = job['error_message']
    
    return result
