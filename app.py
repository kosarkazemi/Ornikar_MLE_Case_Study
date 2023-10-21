import logging
from flask import Flask, request, jsonify
from src.serving import predict  # Import the predict function from serving.py

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    logger.info('Received request at index.')
    return 'Welcome to the lead conversion prediction API!'

@app.route('/predict_lead_conversion', methods=['POST'])
def predict_lead_conversion():
    payload = request.get_json()

    if not payload or not payload.get('model') or not payload.get('input_data'):
        logger.error('Invalid payload. Model or input data not provided.')
        return jsonify({'error': 'Invalid payload. Model or input data not provided.'}), 400

    model = payload.get('model')
    input_data = payload.get('input_data')

    logger.info('Received prediction request for model: %s', model)
    predictions = predict(model, input_data)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    logger.info('Starting the application.')
    app.run(host='0.0.0.0', port=5000, debug=True)
