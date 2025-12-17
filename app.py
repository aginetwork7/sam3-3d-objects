import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import uuid
from api.fal_3d import FalAPI3D

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

fal_api_3d = FalAPI3D()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/agi7/api/ai/person/generate_objects3d')
def index():
    return render_template('index.html')

@app.route('/agi7/api/ai/person/generate_objects3d/results')
def results():
    """Render results page showing original image and PLY viewer."""
    ply = request.args.get('ply') or request.args.get('ply_url')
    img = request.args.get('img') or request.args.get('original_image')
    return render_template('results.html', ply_url=ply, original_image=img)

@app.route('/upload/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files (original images)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/agi7/api/ai/person/generate_objects3d/upload', methods=['POST'])
def upload_and_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    # Save uploaded image
    original_filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{original_filename}")
    file.save(original_path)
    print(f"Saved file to {original_path}")
    
    # Run 3D reconstruction pipeline
    detected_result = fal_api_3d.predict(original_path)

    # Build response
    if not detected_result or 'ply_url' not in detected_result:
        return jsonify({'error': 'No human detected or reconstruction failed'}), 200

    # Public URL for the original uploaded image
    original_image_url = url_for('uploaded_file', filename=os.path.basename(original_path))

    # For now we only return a single result; structure as an array for future multi-person support
    results = [{
        'ply_url': detected_result.get('ply_url')
        # 'glb_url': detected_result.get('glb_url')  # add later if available
    }]

    return jsonify({
        'original_image': original_image_url,
        'results': results
    }), 200

if __name__ == '__main__':
    app.run(debug=False)