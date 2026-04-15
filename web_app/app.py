import os
from flask import Flask
from flask_migrate import Migrate
from .models import db
from .routes import api_bp
from dotenv import load_dotenv
from pathlib import Path

# Define directories
project_dir = Path(__file__).resolve().parent.parent
config_dir = project_dir / "config"
env_path = config_dir / ".env"

# Load environment variables
load_dotenv(dotenv_path=env_path)

# Set Neon url environmental variable name
NEON_DB_URL_ENV_VAR = "NEON_DATABASE_URL"

migrate = Migrate()


def create_app(db_url_env_var: str = NEON_DB_URL_ENV_VAR):
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    uri = os.getenv(db_url_env_var)
    if uri is None:
        raise KeyError(f"Environment variable {db_url_env_var}")
    app.config["SQLALCHEMY_DATABASE_URI"] = uri.replace("postgres://", "postgresql://")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Ensure new connection is made
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from . import models

    app.register_blueprint(api_bp)

    return app


# For development
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
