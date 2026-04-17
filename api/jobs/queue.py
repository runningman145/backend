"""
Job queue management.
Thread-based job queue for processing video detection tasks.
"""
import threading
from datetime import datetime
from flask import current_app
from ..db import get_db
from .worker import process_job


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
                    process_job(job)
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


# Global job queue instance
_job_queue = None


def get_job_queue():
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(num_workers=2)
    return _job_queue
