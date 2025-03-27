from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
from urllib.parse import urlparse
from test import extract_feature
from utils import create_model
import numpy as np
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = create_model()
model.load_weights("results/model.h5")

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return save_path

@app.route('/predict', methods=['GET','POST'])
def predict_gender():
    print("> predict_gender()")
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    if not is_valid_url(url):
        return jsonify({'error': 'Invalid URL provided'}), 400
    
    try:
        # Generate a temporary filename
        filename = secure_filename(os.path.basename(urlparse(url).path))
        if not filename:
            filename = 'audio.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Download the file
        download_file(url, filepath)
        
        if not allowed_file(filename):
            os.remove(filepath)
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Extract features and make prediction
        features = extract_feature(filepath, mel=True).reshape(1, -1)
        male_prob = float(model.predict(features)[0][0])
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        
        # Clean up the downloaded file
        os.remove(filepath)
        
        return jsonify({
            'gender': gender,
            'probabilities': {
                'male': round(male_prob * 100, 2),
                'female': round(female_prob * 100, 2)
            }
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to download file: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True) 
