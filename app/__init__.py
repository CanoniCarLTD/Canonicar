from flask import Flask, render_template
from db import init_db, close_db
from app.controllers.carla_controller import carla_controller


def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")

    # Initialize database
    if init_db():
        print("Successfully connected to the database.")

    # @app.route('/')
    # def index():
    #     return render_template('index.html')

    # Register blueprints
    app.register_blueprint(carla_controller, url_prefix="/carla")

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        close_db()

    return app
