from flask import Flask, render_template
from app.utils.database import init_db
from app.controllers.carla_controller import carla_controller

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # Initialize database
    ans = input("Initailte database? (y/n): ")
    if ans.lower() == 'y':
        init_db(app)

    @app.route('/')
    def index():
        return render_template('index.html')

    # Register blueprints
    app.register_blueprint(carla_controller, url_prefix='/carla')

    return app
