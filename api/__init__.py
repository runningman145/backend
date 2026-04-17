import os

from flask import Flask

from . import db
from . import extensions
from .routes import register_routes
from .ml.loader import load_models
from .jobs.queue import get_job_queue


def create_app(test_config=None):
    """Create and configure the Flask application."""
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'system.sqlite'),
        REID_MODEL_PATH=os.getenv('REID_MODEL_PATH'),  # Path to trained ReID model
    )

    if os.getenv("DATABASE"):
        app.config["DATABASE"] = os.getenv("DATABASE")

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    # Initialize extensions
    extensions.init_extensions(app)

    # Initialize database
    db.init_app(app)

    # Register all routes
    register_routes(app)

    # Register ML model routes
    from .ml import routes as ml_routes
    app.register_blueprint(ml_routes.bp)

    # Register tracking API routes
    from .tracking import routes as tracking_routes
    app.register_blueprint(tracking_routes.bp)

    # Load ML models once on app startup (before first request)
    models_loaded = False

    @app.before_request
    def initialize_on_first_request():
        """Initialize models and job queue on first request."""
        nonlocal models_loaded
        
        if models_loaded:
            return
        
        # Load ML models
        try:
            load_models()
            app.logger.info("ML models loaded successfully")
        except Exception as e:
            app.logger.error(f"Failed to load ML models: {str(e)}")
        
        # Start job queue
        try:
            job_queue = get_job_queue()
            if not job_queue.running:
                job_queue.start(app)
                app.logger.info("Job queue started")
        except Exception as e:
            app.logger.error(f"Failed to start job queue: {str(e)}")
        
        models_loaded = True

    return app