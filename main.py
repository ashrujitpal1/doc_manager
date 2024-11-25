from flask import Flask
import logging
from doc_manager.api.routes import bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api')
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5005)
