"""Flask web application for Veri-Car demo."""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    
    # Simulate analysis (we'll replace this with real model later)
    results = {
        'is_ood': np.random.random() > 0.7,
        'vehicle': {
            'make': 'Toyota',
            'model': 'Camry',
            'type': 'Sedan',
            'year': '2019',
            'confidence': 0.94
        },
        'color': {
            'prediction': 'Silver',
            'confidence': 0.89
        }
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)