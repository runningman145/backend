"""
Shared Flask extensions and objects.
Centralized initialization of extensions like CORS.
"""
from flask_cors import CORS

# CORS will be initialized in app factory
cors = None


def init_extensions(app):
    """Initialize all extensions with the Flask app."""
    global cors
    cors = CORS(app)
