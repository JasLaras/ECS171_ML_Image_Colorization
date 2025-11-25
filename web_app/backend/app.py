"""
Flask Backend for Image Colorization
"""

from flask import Flask, send_from_directory, request, jsonify
from PIL import Image
import torch
import numpy as np
import io
import os
import base64

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import model from backend package
from model import BaseColorizationCNN, HybridColorizationCNN, load_model, preprocess_image, compute_orthonormal_basis, reconstruct_color, LUMINANCE_WEIGHTS, MODEL_HEIGHT, MODEL_WIDTH

# Get the path to the frontend folder
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__, static_folder=frontend_path, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Directory containing checkpoints
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# In-memory map of loaded models: name -> { model, path, loaded, error }
loaded_models = {}


# Compute color basis on startup
w = LUMINANCE_WEIGHTS.to(device)
u1, u2 = compute_orthonormal_basis(w, device)
print(f"Color basis computed: w={w.tolist()}, u1={u1.tolist()}, u2={u2.tolist()}")


def pil_image_to_base64_png(image_pil):
    buf = io.BytesIO()
    image_pil.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return 'data:image/png;base64,' + b64


def load_all_models():
    """Scan `models/` and attempt to load all .pth/.pt files into memory."""
    if not os.path.isdir(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}")
        return

    for fname in sorted(os.listdir(MODELS_DIR)):
        if not (fname.endswith('.pth') or fname.endswith('.pt')):
            continue
        path = os.path.join(MODELS_DIR, fname)
        model_entry = {'path': path, 'loaded': False, 'error': None, 'model': None}
        try:
            # Heuristic: choose architecture by filename suffix or substring.
            lower = fname.lower()
            if 'hybrid' in lower or lower.endswith('_hybrid.pth') or lower.endswith('_hybrid.pt'):
                model_class = HybridColorizationCNN
            elif 'base' in lower or lower.endswith('base.pth') or lower.endswith('base.pt'):
                model_class = BaseColorizationCNN
            else:
                # Try base first, fallback to hybrid
                try:
                    m = load_model(BaseColorizationCNN, path, device)
                    model_entry['model'] = m
                    model_entry['loaded'] = True
                    print(f"Loaded model {fname} as BaseColorizationCNN -> {path}")
                    loaded_models[fname] = model_entry
                    continue
                except Exception:
                    model_class = HybridColorizationCNN

            m = load_model(model_class, path, device)
            model_entry['model'] = m
            model_entry['loaded'] = True
            print(f"Loaded model {fname} -> {path} ({model_class.__name__})")
        except Exception as e:
            model_entry['error'] = str(e)
            print(f"Failed to load model {fname}: {e}")

        loaded_models[fname] = model_entry


# Load all checkpoints at startup
load_all_models()


def postprocess_output(L_tensor, alpha_beta_tensors, image_size=(MODEL_HEIGHT, MODEL_WIDTH)):
    """
    Reconstruct RGB image from luminance and alpha/beta coefficients.
    
    The model outputs (alpha, beta) coefficients in an orthonormal color basis.
    We reconstruct the full RGB image using:
        V = (Y / (w·w)) * w + alpha * u1 + beta * u2
    
    Args:
        L_tensor: Input luminance (1, 1, H, W)
        alpha_beta_tensors: Tuple of (alpha, beta) from model, each (1, 1, H, W)
        image_size: (H, W) tuple for reshaping
        
    Returns:
        PIL.Image: Colorized RGB image
    """
    alpha_pred, beta_pred = alpha_beta_tensors
    
    # Extract and move to CPU
    L = L_tensor.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    alpha = alpha_pred.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    beta = beta_pred.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    
    # Flatten for reconstruction
    L_flat = L.reshape(-1)  # (H*W,)
    alpha_flat = alpha.reshape(-1)  # (H*W,)
    beta_flat = beta.reshape(-1)  # (H*W,)
    
    # Reconstruct colors using orthonormal basis
    V = reconstruct_color(L_flat, alpha_flat, beta_flat, w, u1, u2)  # (H*W, 3)
    
    # Clamp to valid RGB range and reshape
    V = V.clamp(0, 1)  # (H*W, 3)
    rgb_array = V.reshape(image_size[0], image_size[1], 3).numpy() * 255  # (H, W, 3)
    rgb_array = rgb_array.astype(np.uint8)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(rgb_array, mode='RGB')
    return image_pil


@app.route('/')
def index():
    """Serve home page"""
    return send_from_directory(frontend_path, 'project.html')


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files from frontend folder"""
    return send_from_directory(frontend_path, filename)


@app.route('/colorize', methods=['POST'])
def colorize():
    """
    Colorize an image.
    
    Accepts: image file (any size, any color format)
    Returns: colorized RGB image
    """
    # Check if image file is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read requested model names
    requested = request.form.getlist('models')
    if not requested:
        return jsonify({'error': 'No model names provided. Include form field "models".'}), 400

    # Open image
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess image
    tensor_input = preprocess_image(image).to(device)

    results = []

    # Run each requested model (use preloaded models)
    for mname in requested:
        entry = loaded_models.get(mname)
        if entry is None:
            return jsonify({'error': f'Model not found: {mname}'}), 404
        if not entry.get('loaded'):
            return jsonify({'error': f'Model failed to load: {mname}', 'detail': entry.get('error')}), 500

        model_obj = entry['model']

        try:
            with torch.no_grad():
                alpha_beta_output = model_obj(tensor_input)

            colorized_image = postprocess_output(tensor_input, alpha_beta_output, (MODEL_HEIGHT, MODEL_WIDTH))
            img_b64 = pil_image_to_base64_png(colorized_image)
            results.append({'name': mname, 'image_b64': img_b64})
        except Exception as e:
            return jsonify({'error': f'Inference failed for model {mname}: {e}'}), 500

    # Also return the preprocessed grayscale original as a data-uri
    # Convert the preprocessed luminance back to a PIL grayscale image
    L_cpu = tensor_input.detach().cpu().squeeze(0).squeeze(0).numpy()  # (H,W)
    L_img = Image.fromarray((L_cpu * 255).astype('uint8'), mode='L').resize((MODEL_WIDTH, MODEL_HEIGHT))
    original_b64 = pil_image_to_base64_png(L_img.convert('RGB'))

    return jsonify({'original': original_b64, 'results': results}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    loaded = [name for name, e in loaded_models.items() if e.get('loaded')]
    failed = {name: e.get('error') for name, e in loaded_models.items() if not e.get('loaded')}
    return jsonify({
        'status': 'ok',
        'models_loaded': len(loaded),
        'loaded_models': loaded,
        'failed_models': failed
    })



@app.route('/models', methods=['GET'])
def list_models():
    """Return list of available models and their load status."""
    out = []
    for name, e in loaded_models.items():
        out.append({'name': name, 'loaded': bool(e.get('loaded')), 'error': e.get('error')})
    return jsonify({'models': out}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
