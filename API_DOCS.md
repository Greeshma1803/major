# 🌍 EuroSAT Land Use Classification API

A RESTful API for classifying satellite images using deep learning with EfficientNetB0.

## 🚀 Quick Start

### Base URL
```
http://localhost:5000
```

### Health Check
```bash
curl http://localhost:5000/api/health
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/api/health`

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "service": "EuroSAT Land Use Classification API",
  "timestamp": "2024-01-15T10:30:00",
  "model": {
    "status": "loaded",
    "name": "EfficientNetB0",
    "input_shape": [64, 64, 3],
    "num_classes": 10,
    "classes": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
  },
  "version": "1.0.0"
}
```

### 2. Single Image Prediction
**POST** `/api/predict`

Perform SAM segmentation on an uploaded image and classify each segmented region with EfficientNet. If SAM is not available or the checkpoint is missing, the endpoint falls back to classifying the full image.

The response also includes environmental indices such as NDVI for vegetation health assessment.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` field containing image
- Optional: `prompts` field with comma-separated text prompts for segmentation (e.g. `trees,residential,road`)

**Supported Formats:** PNG, JPG, JPEG, TIFF, BMP

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "class": "Residential",
      "class_id": 7,
      "confidence": 95.67,
      "bbox": [34, 12, 120, 94],
      "mask_score": 0.92
    },
    {
      "class": "Forest",
      "class_id": 1,
      "confidence": 88.20,
      "bbox": [8, 100, 128, 200],
      "mask_score": 0.84
    }
  ],
  "environmental_indices": {
    "ndvi": 0.32
  },
  "total_segments": 2,
  "timestamp": "2024-01-15T10:30:00",
  "model_info": {
    "segmentation": "SAM",
    "classification": "EfficientNetB0",
    "input_shape": [64, 64, 3],
    "classes": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
  }
}
```

### 3. Image Annotation (Sliding Window)
**POST** `/api/annotate`

Perform sliding window analysis on large images with visual annotation. The response includes environmental indices such as NDVI alongside annotated predictions.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` field containing image

**Supported Formats:** PNG, JPG, JPEG, TIFF, BMP

**Response:**
```json
{
  "success": true,
  "data": {
    "top_prediction": {
      "class": "Residential",
      "class_id": 7,
      "confidence": 87.45,
      "confidence_raw": 0.8745
    },
    "class_scores": {
      "Highway": 0.0,
      "AnnualCrop": 23.1,
      "Industrial": 0.0,
      "River": 0.0,
      "Residential": 87.5
    },
    "environmental_indices": {
      "ndvi": 0.38
    },
    "annotated_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "image_metadata": {
      "width": 512,
      "height": 512,
      "patches_analyzed": 225
    }
  },
  "timestamp": "2024-01-15T10:30:00",
  "model_info": {
    "name": "EfficientNetB0",
    "input_shape": [64, 64, 3],
    "classes": ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
  }
}
```

## 🔧 Usage Examples

### Python Requests
```python
import requests

# Health check
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Single prediction
with open('satellite_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['data']['class']} ({result['data']['confidence']}%)")

# Annotation
with open('large_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/annotate', files=files)
    result = response.json()
    
    # Save annotated image
    import base64
    img_data = base64.b64decode(result['data']['annotated_image'])
    with open('annotated_image.png', 'wb') as img_file:
        img_file.write(img_data)
```

### cURL Examples
```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST -F "file=@satellite_image.jpg" -F "prompts=trees,residential,road" http://localhost:5000/api/predict

# Annotation
curl -X POST -F "file=@large_image.jpg" http://localhost:5000/api/annotate
```

### JavaScript/Fetch
```javascript
// Health check
fetch('http://localhost:5000/api/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Single prediction
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('prompts', 'trees,residential,road');

fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📊 Land Use Classes

| Class | Description | Color Code |
|-------|-------------|------------|
| AnnualCrop | Annual agricultural crops | Red |
| Forest | Forested areas | Green |
| HerbaceousVegetation | Grasslands and meadows | Light Blue |
| Highway | Roads and highways | Yellow |
| Industrial | Industrial areas | Magenta |
| Pasture | Grazing lands | Cyan |
| PermanentCrop | Permanent crops (orchards, vineyards) | Light Purple |
| Residential | Urban residential areas | Olive |
| River | Water bodies and rivers | Teal |
| SeaLake | Oceans and lakes | Light Pink |

## ⚙️ Technical Details

### Model Architecture
- **Base Model:** EfficientNetB0
- **Input Shape:** 64x64x3 (RGB)
- **Output:** 10 classes with softmax probabilities
- **Framework:** TensorFlow/Keras

### Image Processing
- **Patch Size:** 64x64 pixels
- **Stride:** 32 pixels (for sliding window)
- **Normalization:** Pixel values scaled to [0,1]
- **Format:** RGB (converted from BGR if needed)

### Performance
- **Batch Processing:** Single image inference
- **Memory:** Optimized for large images
- **GPU Support:** Compatible with TensorFlow GPU

## 🚨 Error Handling

### Common Error Responses

**File Not Provided:**
```json
{
  "error": "No file provided",
  "message": "Please include an image file in the request"
}
```

**Invalid File Type:**
```json
{
  "error": "Invalid file type",
  "message": "Please upload an image file (PNG, JPG, JPEG, TIFF, BMP)"
}
```

**Processing Error:**
```json
{
  "success": false,
  "error": "Error details",
  "message": "Failed to process image"
}
```

## 🔒 Security & Limitations

- **File Size:** No explicit limit (limited by server memory)
- **Rate Limiting:** Not implemented (add if needed for production)
- **Authentication:** Not implemented (add if needed for production)
- **CORS:** Enabled for cross-origin requests

## 🚀 Deployment

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Production
```bash
# Use production WSGI server
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or with Docker
docker build -t eurosat-api .
docker run -p 5000:5000 eurosat-api
```

### Environment Variables
- `FLASK_ENV`: Set to `production` for production
- `TF_ENABLE_ONEDNN_OPTS`: Set to `0` to disable oneDNN optimizations

## 📈 Monitoring

### Health Check Endpoint
Monitor `/api/health` for:
- Service availability
- Model loading status
- Response times

### Logs
Check server logs for:
- Request processing times
- Error details
- Model inference performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check the error responses
- Verify image format and size
- Ensure model file is present
- Check server logs for detailed errors
