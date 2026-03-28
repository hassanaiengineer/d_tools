from flask import Flask
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')

    # Register the blueprint
    from app.dashboard_routes import dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix="/")

    return app
