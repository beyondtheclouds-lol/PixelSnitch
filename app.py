"""
Flask web application for deepfake detection
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import torch

from src.models.model_loader import load_model, predict_image
from src.utils.image_processing import preprocess_image, validate_image, detect_faces

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_trained_model():
    """Load the trained model"""
    global model
    model_path = 'saved_models/best_model.pth'
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, device=device, model_name='efficientnet-b0')
            print(f"Model loaded successfully on {device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(f"Model not found at {model_path}. Please train the model first.")
        return False


def cleanup_old_uploads(max_files=20):
    """Keep only the latest N files in uploads folder"""
    upload_dir = app.config['UPLOAD_FOLDER']
    
    if not os.path.exists(upload_dir):
        return
    
    # Get all files with their modification times
    files = []
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if os.path.isfile(filepath):
            mtime = os.path.getmtime(filepath)
            files.append((mtime, filepath, filename))
    
    # Sort by modification time (newest first)
    files.sort(reverse=True)
    
    # Delete files beyond the limit
    if len(files) > max_files:
        for mtime, filepath, filename in files[max_files:]:
            try:
                os.remove(filepath)
                print(f"Deleted old upload: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


def save_prediction_history(prediction_data):
    """Save prediction to history file"""
    history_file = 'history/predictions.json'
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new prediction
    history.append(prediction_data)
    
    # Keep only last 20 predictions
    if len(history) > 20:
        history = history[-20:]
    
    # Save history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Validate image
    is_valid, error_msg = validate_image(filepath)
    if not is_valid:
        os.remove(filepath)
        return jsonify({'error': error_msg}), 400
    
    # Clean up old uploads (keep only latest 20)
    cleanup_old_uploads(max_files=20)
    
    # Optional: Check for faces in the image (informational only)
    # Note: The model was trained on general deepfake images, not just faces
    has_face, num_faces, face_error = detect_faces(filepath)
    
    # Load model if not already loaded
    if model is None:
        if not load_trained_model():
            return jsonify({
                'error': 'Model not available. Please train the model first.',
                'model_path': 'saved_models/best_model.pth'
            }), 503
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(filepath)
        
        # Make prediction
        prediction = predict_image(model, image_tensor, device=device)
        
        # Prepare response data
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'filepath': unique_filename,
            'prediction': prediction['prediction'],
            'is_fake': prediction['is_fake'],
            'confidence': prediction['confidence'],
            'real_probability': prediction['real_probability'],
            'fake_probability': prediction['fake_probability'],
            'has_face': has_face,  # Informational only
            'num_faces': num_faces  # Informational only
        }
        
        # Save to history
        save_prediction_history(prediction_data)
        
        return jsonify(prediction_data)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/history')
def history():
    """Display prediction history"""
    history_file = 'history/predictions.json'
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            predictions = json.load(f)
        # Reverse to show most recent first
        predictions.reverse()
    else:
        predictions = []
    
    return render_template('history.html', predictions=predictions)


@app.route('/api/history')
def api_history():
    """API endpoint for prediction history"""
    history_file = 'history/predictions.json'
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            predictions = json.load(f)
        predictions.reverse()
    else:
        predictions = []
    
    return jsonify(predictions)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Try to load model on startup
    print("Starting Flask app...")
    print(f"Using device: {device}")
    load_trained_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

