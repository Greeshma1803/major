# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.activations import swish
from efficientnet.tfkeras import EfficientNetB0  
from collections import defaultdict
import base64
import io
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # OpenAI SDK is optional at runtime

def build_model_architecture() -> tf.keras.Model:
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(64, 64, 3))
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(64, 64, 3)),
        base_model,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

def load_eurosat_model(model_path: str) -> tf.keras.Model:
    # Try loading full SavedModel with custom objects first
    try:
        try:
            # Prefer the efficientnet package's swish/FixedDropout if available
            try:
                from efficientnet import model as effnet_model
                custom_objects = {
                    'swish': getattr(effnet_model, 'swish', tf.nn.swish),
                    'FixedDropout': getattr(effnet_model, 'FixedDropout', tf.keras.layers.Dropout),
                    'EfficientNetB0': EfficientNetB0,
                }
            except Exception:
                custom_objects = {
                    'swish': tf.nn.swish,
                    'EfficientNetB0': EfficientNetB0,
                }
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception:
            # Fallback: rebuild architecture and load weights
            model = build_model_architecture()
            try:
                model.load_weights(model_path)
                return model
            except Exception:
                # Last resort: try loading without any custom objects
                return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

model = load_eurosat_model('model.keras')

def build_unet(input_shape=(256, 256, 6)):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(b)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D()(c3)
    u2 = tf.keras.layers.concatenate([u2, c1])
    c4 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return tf.keras.Model(inputs, outputs)
unet_model = build_unet()

# Try to load trained U-Net weights if available
try:
    unet_model.load_weights('unet_change_detection.h5')
    print("Loaded trained U-Net weights for change detection")
    UNET_TRAINED = True
except:
    print("No trained U-Net weights found, using image differencing fallback")
    UNET_TRAINED = False
def preprocess_for_unet(img1, img2):
    img1 = cv2.resize(img1, (256, 256)) / 255.0
    img2 = cv2.resize(img2, (256, 256)) / 255.0

    combined = np.concatenate([img1, img2], axis=-1)  # (H,W,6)
    return np.expand_dims(combined, axis=0)
CLASS_NAMES = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def compute_ndvi_rgb(image: np.ndarray) -> float:
    """Approximate NDVI from RGB imagery.

    This uses an RGB proxy for vegetation health when NIR is unavailable.
    The formula is (G - R) / (G + R) computed over the image.
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0

    rgb = image.astype(np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    denom = r + g
    ndvi = np.where(denom == 0, 0.0, (g - r) / denom)
    ndvi_value = float(np.nanmean(ndvi))
    return float(np.clip(ndvi_value, -1.0, 1.0))


def _compute_diff_change_map(image1, image2):
    img1 = cv2.resize(image1, (256, 256))
    img2 = cv2.resize(image2, (256, 256))

    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    thresh_value, _ = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh_value == 0 or thresh_value == 255:
        normalized = diff_gray.astype(np.float32) / 255.0
        thresh_value = int(np.clip(np.mean(normalized) * 255.0 * 1.5, 10, 80))

    change_map = (diff_gray > thresh_value).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    change_map = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    change_map = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)

    change_percentage = np.mean(change_map) * 100
    visualization = img2.copy()
    visualization[change_map == 1] = [255, 0, 0]

    return change_map, change_percentage, visualization


def _mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _is_valid_change_map(change_map, min_pct=0.01, max_pct=0.95):
    pct = np.mean(change_map)
    return min_pct <= pct <= max_pct


def detect_changes_unet(image1, image2):
    """
    Change detection using U-Net model.
    Uses trained U-Net if available, otherwise falls back to image differencing.
    """
    global UNET_TRAINED

    diff_map, diff_percentage, diff_visualization = _compute_diff_change_map(image1, image2)

    if not UNET_TRAINED:
        return diff_map, diff_percentage, diff_visualization

    input_tensor = preprocess_for_unet(image1, image2)
    pred = unet_model.predict(input_tensor, verbose=0)[0]
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = np.squeeze(pred, axis=-1)

    change_map = (pred > 0.5).astype(np.uint8)
    change_percentage = np.mean(change_map) * 100

    if not _is_valid_change_map(change_map):
        print(f"UNet prediction invalid (change_percentage={change_percentage:.2f}), using fallback differencing")
        return diff_map, diff_percentage, diff_visualization

    iou = _mask_iou(change_map, diff_map)
    if iou < 0.2 and abs(change_percentage - diff_percentage) > 15:
        print(f"UNet and diff disagree (iou={iou:.2f}, unet={change_percentage:.2f}%, diff={diff_percentage:.2f}%), using fallback differencing")
        return diff_map, diff_percentage, diff_visualization

    visualization = cv2.resize(image2, (256, 256))
    visualization[change_map == 1] = [255, 0, 0]
    return change_map, change_percentage, visualization

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Convert RGB to BGR for OpenCV encoding
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = (image_array * 255).astype(np.uint8)
    
    # Encode to PNG
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64
    img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return img_str


def preprocess_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image_for_change_detection(image_file):
    """Read and preprocess image from file upload for change detection"""
    img_bytes = image_file.read()
    return preprocess_image_bytes(img_bytes)


def is_blank_image(image: np.ndarray, threshold: float = 0.05) -> bool:
    """Check if image is mostly blank/black/grey (invalid data)."""
    if image is None or image.size == 0:
        return True
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Check if image is mostly uniform color (blank)
    std_dev = np.std(gray.astype(np.float32))
    if std_dev < 5:  # Very low variation = blank/uniform
        return True
    
    # Check if image is mostly black or white
    mean_val = np.mean(gray)
    if mean_val < 10 or mean_val > 245:  # Mostly black or white
        return True
    
    return False


def fetch_satellite_image(lat: float, lon: float, date_text: str, width: int = 512, height: int = 512, span_degrees: float = 0.12):
    """Fetch a satellite image from NASA GIBS using latitude, longitude, and date.
    
    Uses multiple layers and fallback strategies to ensure valid data.
    """
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Date must be in YYYY-MM-DD format')

    lat = float(lat)
    lon = float(lon)
    half_span = span_degrees / 2.0
    min_lat = max(-90.0, lat - half_span)
    max_lat = min(90.0, lat + half_span)
    min_lon = max(-180.0, lon - half_span)
    max_lon = min(180.0, lon + half_span)

    # Try multiple layers in order of preference
    layers_to_try = [
        'MODIS_Aqua_CorrectedReflectance_TrueColor',  # Aqua satellite - often has better coverage
        'MODIS_Terra_CorrectedReflectance_TrueColor',  # Terra satellite
        'VIIRS_SNPP_CorrectedReflectance_TrueColor',   # VIIRS satellite
        'BlueMarble_ShadedRelief_Bathymetry',          # Fallback: Blue Marble
    ]

    image = None
    last_error = None

    base_date = datetime.strptime(date_text, '%Y-%m-%d')
    dates_to_try = [base_date - timedelta(days=offset) for offset in range(0, 4)]

    for date_obj in dates_to_try:
        date_str = date_obj.strftime('%Y-%m-%d')
        for layer in layers_to_try:
            try:
                params = {
                    'SERVICE': 'WMS',
                    'VERSION': '1.3.0',
                    'REQUEST': 'GetMap',
                    'LAYERS': layer,
                    'STYLES': '',
                    'FORMAT': 'image/png',
                    'CRS': 'EPSG:4326',
                    'BBOX': f'{min_lat},{min_lon},{max_lat},{max_lon}',
                    'WIDTH': str(width),
                    'HEIGHT': str(height),
                    'TIME': date_str,
                    'TILED': 'FALSE'
                }
                if layer == 'BlueMarble_ShadedRelief_Bathymetry':
                    params.pop('TIME', None)

                url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?' + urllib.parse.urlencode(params)
                print(f"Trying layer {layer} for date {date_str}...")

                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = response.read()

                decoded_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if decoded_img is None:
                    last_error = f'{layer}: Failed to decode image'
                    continue

                # Check if image is valid (not blank)
                rgb_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                if is_blank_image(rgb_img):
                    last_error = f'{layer}: Image is blank or no data available for this date/location'
                    print(f"  {layer} returned blank image, trying next layer...")
                    continue

                print(f"  Successfully fetched from {layer}")
                return rgb_img

            except (HTTPError, URLError, Exception) as e:
                last_error = f'{layer}: {str(e)}'
                print(f"  Failed with {layer}: {str(e)}")
                continue

    # If all layers failed, raise error with guidance
    raise RuntimeError(
        f'Could not fetch valid satellite imagery for these coordinates and date. '
        f'Last error: {last_error}. '
        f'Please try: 1) Different dates (use recent dates), 2) Different location, '
        f'3) Use the image upload option instead.'
    )

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/test_images/<path:filename>')
def serve_test_images(filename: str):
    # Serve images from the test_images directory so they can be referenced in the frontend
    return send_from_directory('test_images', filename)

@app.route('/api')
def api_info():
    return jsonify({
        "message": "EuroSAT Land Use Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "annotate": "/api/annotate",
            "health": "/api/health",
            "llm_query": "/api/llm/query",
            "llm_report": "/api/llm/report"
        },
        "usage": "Send POST requests with image files to /api/predict or /api/annotate"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict land use class for a single image patch.
    
    Request:
        - POST with 'file' field containing image
        
    Response:
        - JSON with predicted class and confidence
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please include an image file in the request'
        }), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Please upload an image file (PNG, JPG, JPEG, TIFF, BMP)'
        }), 400
    
    try:
        image_bytes = file.read()
        original_image = preprocess_image_bytes(image_bytes)
        ndvi_value = compute_ndvi_rgb(original_image)
        img = preprocess_image(image_bytes)
        preds = model.predict(img, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return jsonify({
            'success': True,
            'data': {
                'class': CLASS_NAMES[class_idx],
                'class_id': int(class_idx),
                'confidence': round(confidence * 100, 2),
                'confidence_raw': float(confidence),
                'all_predictions': {
                    CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2) 
                    for i in range(len(CLASS_NAMES))
                },
                'environmental_indices': {
                    'ndvi': round(ndvi_value, 4)
                }
            },
            'timestamp': str(np.datetime64('now')),
            'model_info': {
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'classes': CLASS_NAMES
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process image'
        }), 500

CLASS_COLORS = {
    'AnnualCrop': (255, 0, 0),      # Red
    'Forest': (0, 255, 0),          # Green
    'HerbaceousVegetation': (173, 216, 230),  # Light Blue
    'Highway': (255, 255, 0),       # Yellow
    'Industrial': (255, 0, 255),    # Magenta
    'Pasture': (0, 255, 255),       # Cyan
    'PermanentCrop': (221, 160, 221), # Light Purple
    'Residential': (128, 128, 0),   # Olive
    'River': (0, 128, 128),         # Teal
    'SeaLake': (255, 182, 193),     # Light Pink
}

def draw_label_with_bg_and_arrow(image, text, position, bg_color, dot_center, font_scale=0.9, thickness=3):
    """Draws a label with a colored background, black text, and an arrow pointing to the label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = position
    bg_x1, bg_y1 = x - 5, y - text_h - 5
    bg_x2, bg_y2 = x + text_w + 5, y + 5

    # Draw background rectangle
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)

    # Draw black text
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    # Draw an arrow from the dot center to the label
    cv2.arrowedLine(image, dot_center, (x + text_w // 2, bg_y1), bg_color, 2)

@app.route('/api/annotate', methods=['POST'])
def annotate():
    """
    Perform sliding window annotation on a large image with comprehensive analysis.
    
    Request:
        - POST with 'file' field containing image
        
    Response:
        - JSON with annotated image, comprehensive analysis, charts data, and metadata
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please include an image file in the request'
        }), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Please upload an image file (PNG, JPG, JPEG, TIFF, BMP)'
        }), 400
    
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        input_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if input_image_bgr is None:
            return jsonify({
                'error': 'Unable to decode image',
                'message': 'The uploaded file could not be processed as an image'
            }), 400
        
        input_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = input_image.shape

        ndvi_value = compute_ndvi_rgb(input_image)

        patch_size = 64
        stride = 32

        # Perform comprehensive analysis
        analysis = analyze_image_patches(input_image, patch_size, stride)
        
        overlay = input_image.copy()
        class_detections = defaultdict(list)

        # Slide over image for visualization
        for y in range(0, max(height - patch_size + 1, 1), stride):
            for x in range(0, max(width - patch_size + 1, 1), stride):
                patch = input_image[y:y + patch_size, x:x + patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue
                patch_float = patch.astype(np.float32) / 255.0
                patch_input = np.expand_dims(patch_float, axis=0)
                prediction = model.predict(patch_input, verbose=0)
                predicted_label = int(np.argmax(prediction, axis=1)[0])
                confidence = float(np.max(prediction)) * 100.0
                class_detections[predicted_label].append((x, y, confidence))

        # Calculate top confidence scores for each class
        output_order = ['Highway', 'AnnualCrop', 'Industrial', 'River', 'Residential']
        scores = {}
        for class_name in output_order:
            if class_name in CLASS_NAMES:
                class_label = CLASS_NAMES.index(class_name)
                if class_label in class_detections and class_detections[class_label]:
                    max_conf = max(d[2] for d in class_detections[class_label])
                    scores[class_name] = round(max_conf, 1)
                else:
                    scores[class_name] = 0.0
            else:
                scores[class_name] = 0.0

        # Visualization: draw best point per detected class
        print(f"Debug: Found {len(class_detections)} classes with detections")
        for class_label, detections in class_detections.items():
            if not detections:
                continue
            best_detection = max(detections, key=lambda d: d[2])
            x, y, confidence = best_detection
            class_name = CLASS_NAMES[class_label]
            color = CLASS_COLORS[class_name]
            print(f"Debug: Drawing {class_name} at ({x}, {y}) with confidence {confidence:.1f}%")

            # Draw a larger dot at the center of the patch
            cx = x + patch_size // 2
            cy = y + patch_size // 2
            cv2.circle(overlay, (cx, cy), 15, color, -1)
            cv2.circle(overlay, (cx, cy), 15, (0, 0, 0), 2)

            # Draw label with background and arrow
            label_text = f"{class_name}: {confidence:.1f}%"
            
            # Smart label positioning
            label_x = cx + 20
            label_y = cy - 20
            
            if label_x + 150 > width:
                label_x = cx - 170
            if label_y < 30:
                label_y = cy + 20
            elif label_y > height - 30:
                label_y = cy - 20
                
            draw_label_with_bg_and_arrow(overlay, label_text, (label_x, label_y), color, (cx, cy))

        # Encode annotated image to base64 PNG
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.png', overlay_bgr)
        if not success:
            print("Failed to encode PNG image")
            return jsonify({
                'error': 'Failed to encode annotated image',
                'message': 'Could not process the annotated image'
            }), 500
        
        b64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')

        # Overall top-1 prediction on center crop
        center_y = max((height - patch_size) // 2, 0)
        center_x = max((width - patch_size) // 2, 0)
        center_patch = input_image[center_y:center_y+patch_size, center_x:center_x+patch_size]
        center_patch = center_patch.astype(np.float32) / 255.0
        center_pred = model.predict(np.expand_dims(center_patch, 0), verbose=0)
        top_idx = int(np.argmax(center_pred, axis=1)[0])
        top_conf = float(np.max(center_pred))

        # Calculate additional insights
        coverage = analysis['coverage_percent']
        counts = analysis['class_counts']
        
        # Land use categories
        vegetation_classes = ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop']
        water_classes = ['River', 'SeaLake']
        urban_classes = ['Residential', 'Industrial', 'Highway']
        
        vegetation_coverage = sum(coverage.get(c, 0.0) for c in vegetation_classes)
        water_coverage = sum(coverage.get(c, 0.0) for c in water_classes)
        urban_coverage = sum(coverage.get(c, 0.0) for c in urban_classes)
        
        # Find dominant classes
        sorted_coverage = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        top_3_classes = sorted_coverage[:3]
        
        # Calculate diversity metrics
        total_patches = analysis['total_patches']
        detected_classes = len([c for c in counts.values() if c > 0])
        diversity_index = detected_classes / len(CLASS_NAMES) * 100
        
        # Prepare chart data
        chart_data = {
            'coverage_percent': coverage,
            'class_counts': counts,
            'categories': {
                'vegetation': {
                    'coverage': round(vegetation_coverage, 2),
                    'classes': vegetation_classes,
                    'count': sum(counts.get(c, 0) for c in vegetation_classes)
                },
                'water': {
                    'coverage': round(water_coverage, 2),
                    'classes': water_classes,
                    'count': sum(counts.get(c, 0) for c in water_classes)
                },
                'urban': {
                    'coverage': round(urban_coverage, 2),
                    'classes': urban_classes,
                    'count': sum(counts.get(c, 0) for c in urban_classes)
                }
            }
        }

        response_data = {
            'success': True,
            'data': {
                'top_prediction': {
                    'class': CLASS_NAMES[top_idx],
                    'class_id': int(top_idx),
                    'confidence': round(top_conf * 100, 2),
                    'confidence_raw': float(top_conf)
                },
                'class_scores': scores,
                'annotated_image': b64_image,
                'environmental_indices': {
                    'ndvi': round(ndvi_value, 4)
                },
                'comprehensive_analysis': {
                    'image_metadata': {
                        'width': width,
                        'height': height,
                        'total_patches': total_patches,
                        'patch_size': patch_size,
                        'stride': stride
                    },
                    'coverage_percent': coverage,
                    'class_counts': counts,
                    'land_use_categories': {
                        'vegetation': {
                            'coverage_percent': round(vegetation_coverage, 2),
                            'patch_count': sum(counts.get(c, 0) for c in vegetation_classes),
                            'classes': vegetation_classes
                        },
                        'water': {
                            'coverage_percent': round(water_coverage, 2),
                            'patch_count': sum(counts.get(c, 0) for c in water_classes),
                            'classes': water_classes
                        },
                        'urban': {
                            'coverage_percent': round(urban_coverage, 2),
                            'patch_count': sum(counts.get(c, 0) for c in urban_classes),
                            'classes': urban_classes
                        }
                    },
                    'insights': {
                        'dominant_class': top_3_classes[0][0] if top_3_classes else 'Unknown',
                        'dominant_coverage': round(top_3_classes[0][1], 2) if top_3_classes else 0.0,
                        'top_3_classes': [{'class': c[0], 'coverage': round(c[1], 2)} for c in top_3_classes],
                        'diversity_index': round(diversity_index, 2),
                        'detected_classes': detected_classes,
                        'total_classes': len(CLASS_NAMES)
                    }
                },
                'chart_data': chart_data
            },
            'timestamp': str(np.datetime64('now')),
            'model_info': {
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'classes': CLASS_NAMES
            }
        }
        
        print(f"Response data keys: {list(response_data['data'].keys())}")
        print(f"Comprehensive analysis included: {'comprehensive_analysis' in response_data['data']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process image annotation'
        }), 500

# =====================================================
# CHANGE DETECTION ENDPOINT
# =====================================================

@app.route('/api/change-detection', methods=['POST'])
def change_detection():
    """
    API endpoint for change detection between two satellite images.
    
    Request:
        - POST with 'image1' (before) and 'image2' (after) fields containing images
        
    Response:
        - JSON with change detection visualization and statistics
    """
    try:
        use_coords = all([request.form.get(k) for k in ('latitude', 'longitude', 'before_date', 'after_date')])

        if use_coords:
            latitude = request.form.get('latitude').strip()
            longitude = request.form.get('longitude').strip()
            before_date = request.form.get('before_date').strip()
            after_date = request.form.get('after_date').strip()

            if not latitude or not longitude or not before_date or not after_date:
                return jsonify({
                    'error': 'Latitude, longitude, before_date and after_date are required',
                    'message': 'Please provide coordinates and both dates for satellite image fetch'
                }), 400

            print(f"Fetching satellite imagery for location {latitude},{longitude}")
            image1 = fetch_satellite_image(latitude, longitude, before_date)
            print(f"Before image loaded: {image1.shape}")
            image2 = fetch_satellite_image(latitude, longitude, after_date)
            print(f"After image loaded: {image2.shape}")
            requested_location = {
                'latitude': float(latitude),
                'longitude': float(longitude),
                'before_date': before_date,
                'after_date': after_date
            }
        else:
            if 'image1' not in request.files or 'image2' not in request.files:
                return jsonify({
                    'error': 'Both image1 and image2 are required',
                    'message': 'Please upload both before and after images or provide coordinates with dates'
                }), 400

            image1_file = request.files['image1']
            image2_file = request.files['image2']

            # Validate file types
            for img_file, name in [(image1_file, 'image1'), (image2_file, 'image2')]:
                if not img_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    return jsonify({
                        'error': f'Invalid file type for {name}',
                        'message': 'Please upload image files (PNG, JPG, JPEG, TIFF, BMP)'
                    }), 400

            print("Starting change detection...")
            print("Loading image 1...")
            image1 = preprocess_image_for_change_detection(image1_file)
            print(f"Image 1 loaded: {image1.shape}")

            print("Loading image 2...")
            image2 = preprocess_image_for_change_detection(image2_file)
            print(f"Image 2 loaded: {image2.shape}")
            requested_location = None

        print("Performing change detection with U-Net model...")
        change_map, change_percentage, visualization = detect_changes_unet(
            image1, image2
        )
        
        print("Building change detection summary...")
        change_summary, change_details = build_change_detection_summary(image1, image2, change_map)
        
        # Convert all images to base64
        print("Encoding images...")
        before_image_base64 = image_to_base64(image1)
        after_image_base64 = image_to_base64(image2)
        change_image_base64 = image_to_base64(visualization)
        print(f"Images encoded. Before: {len(before_image_base64)}, After: {len(after_image_base64)}, Change: {len(change_image_base64)}")
        
        response_data = {
            'success': True,
            'before_image': before_image_base64,
            'after_image': after_image_base64,
            'change_image': change_image_base64,
            'change_percentage': round(change_percentage, 2),
            'change_summary': change_summary,
            'change_details': change_details,
            'requested_location': requested_location,
            'data': {
                'before_image': before_image_base64,
                'after_image': after_image_base64,
                'change_image': change_image_base64,
                'change_percentage': round(change_percentage, 2),
                'total_pixels': int(change_map.size),
                'changed_pixels': int(np.sum(change_map)),
                'image_dimensions': {
                    'width': int(visualization.shape[1]),
                    'height': int(visualization.shape[0])
                },
                'change_summary': change_summary,
                'change_details': change_details,
                'requested_location': requested_location,
            },
            'message': f'Change detection completed. {round(change_percentage, 2)}% of the area has changed.',
            'timestamp': str(np.datetime64('now')),
            'algorithm': {
                'method': 'U-Net CNN' if UNET_TRAINED else 'U-Net with Image Differencing',
                'input_shape': '(256, 256, 6)',
                'trained': UNET_TRAINED,
                'description': 'Deep learning-based change detection using U-Net architecture' if UNET_TRAINED else 'Change detection using image differencing (U-Net model not trained)'
            }
        }
        
        print("Change detection completed successfully!")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in change detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to perform change detection'
        }), 500
def analyze_image_patches(image_rgb: np.ndarray, patch_size: int = 64, stride: int = 32):
    """Run sliding-window classification and compute per-class counts and coverage."""
    height, width, _ = image_rgb.shape
    class_counts = {name: 0 for name in CLASS_NAMES}
    total_patches = 0
    for y in range(0, max(height - patch_size + 1, 1), stride):
        for x in range(0, max(width - patch_size + 1, 1), stride):
            patch = image_rgb[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patch_float = patch.astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(patch_float, 0), verbose=0)
            class_idx = int(np.argmax(pred, axis=1)[0])
            class_name = CLASS_NAMES[class_idx]
            class_counts[class_name] += 1
            total_patches += 1

    coverage = {k: (v / total_patches * 100.0 if total_patches else 0.0) for k, v in class_counts.items()}
    return {
        'width': width,
        'height': height,
        'patch_size': patch_size,
        'stride': stride,
        'total_patches': total_patches,
        'class_counts': class_counts,
        'coverage_percent': coverage,
    }


def build_llm_context_from_analysis(analysis: dict) -> str:
    """Create a concise textual summary from analysis for LLM context."""
    coverage_sorted = sorted(analysis['coverage_percent'].items(), key=lambda kv: kv[1], reverse=True)
    top_lines = [f"{k}: {v:.1f}%" for k, v in coverage_sorted[:5]]
    lines = [
        f"Image size: {analysis['width']}x{analysis['height']} pixels",
        f"Patches analyzed: {analysis['total_patches']} (patch {analysis['patch_size']} stride {analysis['stride']})",
        "Top coverages: " + ", ".join(top_lines)
    ]
    return "\n".join(lines)


def build_llm_context(analysis: dict = None, change_summary: str = None) -> str:
    """Build combined LLM context from classification analysis and change detection summary."""
    parts = []
    if analysis:
        parts.append(build_llm_context_from_analysis(analysis))
    if change_summary:
        parts.append(f"Change detection insights: {change_summary}")
    return "\n\n".join(parts)


def _classify_patch(patch: np.ndarray) -> str:
    patch_resized = cv2.resize(patch, (64, 64))
    patch_float = patch_resized.astype(np.float32) / 255.0
    preds = model.predict(np.expand_dims(patch_float, 0), verbose=0)
    class_idx = int(np.argmax(preds, axis=1)[0])
    return CLASS_NAMES[class_idx]


def build_change_detection_summary(image1: np.ndarray, image2: np.ndarray, change_map: np.ndarray):
    """Generate a human-readable summary and structured detail payload for change detection results."""
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))
    height, width = change_map.shape
    before_counts = defaultdict(int)
    after_counts = defaultdict(int)
    changed_patch_count = 0
    patch_size = 64
    stride = 64

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch_mask = change_map[y:y + patch_size, x:x + patch_size]
            if np.mean(patch_mask) < 0.15:
                continue
            patch1 = image1[y:y + patch_size, x:x + patch_size]
            patch2 = image2[y:y + patch_size, x:x + patch_size]
            if patch1.shape[:2] != (patch_size, patch_size) or patch2.shape[:2] != (patch_size, patch_size):
                continue

            before_cls = _classify_patch(patch1)
            after_cls = _classify_patch(patch2)
            before_counts[before_cls] += 1
            after_counts[after_cls] += 1
            changed_patch_count += 1

    def pct(count_dict, classes):
        return sum(count_dict.get(c, 0) for c in classes) / max(changed_patch_count, 1) * 100.0

    before_urban = pct(before_counts, ['Residential', 'Industrial', 'Highway'])
    after_urban = pct(after_counts, ['Residential', 'Industrial', 'Highway'])
    before_veg = pct(before_counts, ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop'])
    after_veg = pct(after_counts, ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop'])
    before_water = pct(before_counts, ['River', 'SeaLake'])
    after_water = pct(after_counts, ['River', 'SeaLake'])

    statements = []
    if after_urban > before_urban + 10:
        statements.append("More buildings and urbanization detected.")
    elif before_urban > after_urban + 10:
        statements.append("Less urban area detected in the changed region.")

    if after_veg < before_veg - 10:
        statements.append("Less greenery detected, suggesting possible deforestation or vegetation loss.")
    elif after_veg > before_veg + 10:
        statements.append("More vegetation detected in the changed region.")

    if after_water > before_water + 10:
        statements.append("Water bodies appear to have expanded.")
    elif before_water > after_water + 10:
        statements.append("Water bodies appear to have receded.")
    else:
        statements.append("Water body coverage remained relatively stable.")

    if not statements:
        top_before = sorted(before_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        top_after = sorted(after_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        before_desc = ", ".join([f"{k} ({v})" for k, v in top_before])
        after_desc = ", ".join([f"{k} ({v})" for k, v in top_after])
        statements.append(f"Detected change classes before: {before_desc}; after: {after_desc}.")

    summary_text = " ".join(statements)
    summary_details = {
        'changed_patch_count': int(changed_patch_count),
        'before_urban_pct': round(before_urban, 2),
        'after_urban_pct': round(after_urban, 2),
        'before_vegetation_pct': round(before_veg, 2),
        'after_vegetation_pct': round(after_veg, 2),
        'before_water_pct': round(before_water, 2),
        'after_water_pct': round(after_water, 2),
        'statements': statements,
        'before_counts': dict(before_counts),
        'after_counts': dict(after_counts),
    }
    return summary_text, summary_details


def rule_based_answer(question: str, analysis: dict) -> str:
    q = (question or "").strip().lower()
    coverage = analysis['coverage_percent']
    counts = analysis['class_counts']
    # Largest structure detected
    if 'largest' in q or 'most' in q or 'dominant' in q:
        top_cls = max(coverage.items(), key=lambda kv: kv[1])[0] if coverage else 'Unknown'
        return f"The largest/dominant class is {top_cls} at {coverage.get(top_cls, 0.0):.1f}% coverage."
    # Water bodies count / coverage
    if 'water' in q or 'river' in q or 'lake' in q:
        water_classes = ['River', 'SeaLake']
        water_count = sum(counts.get(c, 0) for c in water_classes)
        water_pct = sum(coverage.get(c, 0.0) for c in water_classes)
        return f"Estimated water patches: {water_count} (~{water_pct:.1f}% coverage) across {', '.join(water_classes)}."
    # Vegetation percentage
    if 'vegetation' in q or 'forest' in q or 'crop' in q or 'pasture' in q:
        veg_classes = ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop']
        veg_pct = sum(coverage.get(c, 0.0) for c in veg_classes)
        return f"Estimated vegetation coverage is ~{veg_pct:.1f}% (classes: {', '.join(veg_classes)})."
    # Industrial/urban
    if 'urban' in q or 'residential' in q or 'industrial' in q:
        urban_classes = ['Residential', 'Industrial', 'Highway']
        urban_pct = sum(coverage.get(c, 0.0) for c in urban_classes)
        return f"Estimated urban/transport coverage is ~{urban_pct:.1f}% ({', '.join(urban_classes)})."
    # Default: return top 3
    top3 = sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return "Top classes: " + ", ".join([f"{k} {v:.1f}%" for k, v in top3])


def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and OpenAI is not None:
        return OpenAI(api_key=api_key)
    return None


@app.route('/api/llm/query', methods=['POST'])
def llm_query():
    """Answer natural language questions about the uploaded image or change detection summary using LLM, with offline fallback."""
    question = request.form.get('question', '').strip()
    change_summary = request.form.get('change_summary', '').strip()
    file = request.files.get('file')

    if not file and not change_summary:
        return jsonify({'error': 'No file or change summary provided'}), 400

    analysis = None
    if file:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return jsonify({'error': 'Invalid file type'}), 400
        nparr = np.frombuffer(file.read(), np.uint8)
        input_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if input_image_bgr is None:
            return jsonify({'error': 'Unable to decode image'}), 400
        image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        analysis = analyze_image_patches(image_rgb)

    if not question:
        question = 'Summarize the available image and change detection findings.'

    client = get_openai_client()
    context = build_llm_context(analysis=analysis, change_summary=change_summary)

    if client is None:
        if analysis:
            answer = rule_based_answer(question, analysis)
        else:
            answer = f"Change detection findings: {change_summary}"
    else:
        try:
            prompt = (
                "You are a helpful remote sensing analyst."
                " Answer the user's question strictly based on the provided context."
                " If insufficient, say you are uncertain.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            completion = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "You analyze satellite land use outputs concisely and factually."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            answer = completion.choices[0].message.content.strip()
        except Exception:
            answer = rule_based_answer(question, analysis) if analysis else f"Change detection findings: {change_summary}"

    return jsonify({
        'success': True,
        'data': {
            'question': question,
            'answer': answer,
            'analysis': analysis,
            'change_summary': change_summary,
        },
        'timestamp': str(np.datetime64('now')),
    })


@app.route('/api/llm/report', methods=['POST'])
def llm_report():
    """Generate a prose report summarizing coverage, notable findings, and change detection insights."""
    change_summary = request.form.get('change_summary', '').strip()
    file = request.files.get('file')
    date_text = request.form.get('date') or datetime.utcnow().strftime('%Y-%m-%d')

    if not file and not change_summary:
        return jsonify({'error': 'No file or change summary provided'}), 400

    try:
        analysis = None
        if file:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                return jsonify({'error': 'Invalid file type'}), 400
            nparr = np.frombuffer(file.read(), np.uint8)
            input_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if input_image_bgr is None:
                return jsonify({'error': 'Unable to decode image'}), 400
            image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
            analysis = analyze_image_patches(image_rgb)

        context = build_llm_context(analysis=analysis, change_summary=change_summary)
        client = get_openai_client()

        if client is None:
            if analysis:
                cov = analysis['coverage_percent']
                water_pct = cov.get('River', 0.0) + cov.get('SeaLake', 0.0)
                urban_pct = cov.get('Residential', 0.0) + cov.get('Industrial', 0.0) + cov.get('Highway', 0.0)
                veg_pct = sum(cov.get(c, 0.0) for c in ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop'])
                top_cls = max(cov.items(), key=lambda kv: kv[1])[0] if cov else 'Unknown'
                report = (
                    f"This satellite image, taken on {date_text}, shows that approximately "
                    f"{veg_pct:.1f}% of the area is covered by vegetation, "
                    f"{water_pct:.1f}% by water bodies, and {urban_pct:.1f}% by urban/transport infrastructure. "
                    f"The dominant land-use class is {top_cls}."
                )
                if change_summary:
                    report += f" Change detection summary: {change_summary}"
            else:
                report = f"Change detection report: {change_summary}"
        else:
            try:
                prompt = (
                    "Draft a concise analytical report (80-150 words) about a satellite image."
                    " Use only the provided context. Include percentages, dominant classes, and any change detection findings."
                    f"\n\nDate: {date_text}\nContext:\n{context}"
                )
                completion = client.chat.completions.create(
                    model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                    messages=[
                        {"role": "system", "content": "You write clear, neutral remote sensing reports."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=250,
                )
                report = completion.choices[0].message.content.strip()
            except Exception:
                if analysis:
                    cov = analysis['coverage_percent']
                    water_pct = cov.get('River', 0.0) + cov.get('SeaLake', 0.0)
                    urban_pct = cov.get('Residential', 0.0) + cov.get('Industrial', 0.0) + cov.get('Highway', 0.0)
                    veg_pct = sum(cov.get(c, 0.0) for c in ['Forest', 'HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'AnnualCrop'])
                    top_cls = max(cov.items(), key=lambda kv: kv[1])[0] if cov else 'Unknown'
                    report = (
                        f"This satellite image, taken on {date_text}, shows that approximately "
                        f"{veg_pct:.1f}% of the area is covered by vegetation, "
                        f"{water_pct:.1f}% by water bodies, and {urban_pct:.1f}% by urban/transport infrastructure. "
                        f"The dominant land-use class is {top_cls}."
                    )
                    if change_summary:
                        report += f" Change detection summary: {change_summary}"
                else:
                    report = f"Change detection report: {change_summary}"

        return jsonify({
            'success': True,
            'data': {
                'date': date_text,
                'report': report,
                'analysis': analysis,
                'change_summary': change_summary,
            },
            'timestamp': str(np.datetime64('now')),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-image', methods=['GET'])
def test_image():
    """
    Test endpoint to verify image processing and base64 encoding.
    """
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :, 0] = 255  # Red channel
        
        # Add some text
        cv2.putText(test_img, 'TEST', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode to base64
        success, buffer = cv2.imencode('.png', test_img)
        if not success:
            return jsonify({'error': 'Failed to encode test image'}), 500
        
        b64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'test_image': b64_image,
            'message': 'Test image generated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    
    Response:
        - JSON with service status and model info
    """
    try:
        # Test model with a dummy input
        dummy_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        return jsonify({
            'status': 'healthy',
            'service': 'EuroSAT Land Use Classification API',
            'timestamp': str(np.datetime64('now')),
            'model': {
                'status': 'loaded',
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'num_classes': len(CLASS_NAMES),
                'classes': CLASS_NAMES
            },
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': str(np.datetime64('now'))
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
