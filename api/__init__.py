import os

from flask import Flask
from flask_cors import CORS


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'system.sqlite'),
        REID_MODEL_PATH=os.getenv('REID_MODEL_PATH'),  # Path to trained ReID model
    )
    
    # enable CORS for all routes
    CORS(app)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    # a simple page that says hello
    # @app.route('/hello')
    # def hello():
    #     return 'Hello, World!'
    from . import db
    db.init_app(app)

    from . import tracking
    app.register_blueprint(tracking.bp)

    from . import cameras
    app.register_blueprint(cameras.bp)

    from . import model
    app.register_blueprint(model.bp)
    
    from . import tracking_endpoints
    app.register_blueprint(tracking_endpoints.bp)
    
    from . import video_endpoints
    app.register_blueprint(video_endpoints.bp)
    
    # Load ML models on app startup
    @app.before_first_request
    def load_models():
        try:
            from . import model as model_module
            model_module._load_models()
            app.logger.info("ML models loaded successfully")
        except Exception as e:
            app.logger.error(f"Failed to load ML models: {str(e)}")
    
    # Start background job queue
    @app.before_first_request
    def start_job_queue():
        try:
            from . import jobs as jobs_module
            job_queue = jobs_module.get_job_queue()
            job_queue.start(app)
            app.logger.info("Job queue started")
        except Exception as e:
            app.logger.error(f"Failed to start job queue: {str(e)}")

    return app