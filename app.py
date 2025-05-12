from flask import Flask
from flask_cors import CORS # type: ignore
from scraping import scraping_bp
from predict import predict_bp
from forcast import forcast_bp
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Register Blueprints for modular structure
app.register_blueprint(scraping_bp, url_prefix='/scraping')
app.register_blueprint(predict_bp, url_prefix='/predict')
app.register_blueprint(forcast_bp, url_prefix='/forcast')


# Debugging
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

@app.route('/')
def hello_world():
    return "Backend Running Successfully"

if __name__ == '__main__':
    app.run(debug=True)