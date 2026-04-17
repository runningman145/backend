"""
Routes module - contains all API endpoint blueprints.
Each file handles routes for a specific resource.
"""
from . import health, cameras, detections, uploads, videos, jobs


def register_routes(app):
    """Register all route blueprints with the Flask app."""
    app.register_blueprint(health.bp)
    app.register_blueprint(cameras.bp)
    app.register_blueprint(detections.bp)
    app.register_blueprint(uploads.bp)
    app.register_blueprint(videos.bp)
    app.register_blueprint(jobs.bp)
