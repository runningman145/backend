"""
Health check endpoints.
Provides liveness, readiness, and status checks.
"""
from flask import Blueprint, jsonify
from ..db import get_db

bp = Blueprint('health', __name__, url_prefix='/health')


@bp.route('', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Backend API'
    }), 200


@bp.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check - verifies database connectivity."""
    try:
        db = get_db()
        db.execute('SELECT 1').fetchone()
        
        return jsonify({
            'status': 'ready',
            'database': 'connected',
            'service': 'Backend API'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'database': 'disconnected',
            'error': str(e)
        }), 503


@bp.route('/live', methods=['GET'])
def liveness_check():
    """Liveness check - verifies service is running."""
    return jsonify({
        'status': 'alive',
        'service': 'Backend API'
    }), 200


@bp.route('/status', methods=['GET'])
def detailed_status():
    """Detailed status information."""
    try:
        db = get_db()
        
        # Get database stats
        cameras_count = db.execute('SELECT COUNT(*) as count FROM cameras').fetchone()['count']
        detections_count = db.execute('SELECT COUNT(*) as count FROM detections').fetchone()['count']
        
        # Check database connectivity
        db.execute('SELECT 1').fetchone()
        db_status = 'connected'
        
        return jsonify({
            'status': 'healthy',
            'service': 'Backend API',
            'database': {
                'status': db_status,
                'stats': {
                    'cameras': cameras_count,
                    'detections': detections_count,
                }
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'Backend API',
            'error': str(e)
        }), 503


@bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint."""
    return jsonify({'pong': True}), 200
