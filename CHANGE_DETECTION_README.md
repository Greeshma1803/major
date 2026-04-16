# Land Use Change Detection with U-Net

This application implements change detection between satellite images using a U-Net convolutional neural network.

## Features

- **U-Net Architecture**: Deep learning-based change detection
- **Image Differencing Fallback**: Works even without trained model
- **Web Interface**: Easy-to-use web application
- **REST API**: Programmatic access via `/api/change-detection`

## How It Works

The change detection process:

1. Takes two input images (before and after)
2. Concatenates them into a 6-channel input (RGB + RGB)
3. Feeds through U-Net to predict change mask
4. Returns change map with percentage of changed area

## Training the U-Net Model

For optimal results, train the U-Net model on real change detection data:

```bash
python train_unet.py
```

This will:
- Create synthetic training data (for demonstration)
- Train the U-Net model
- Save weights as `unet_change_detection.h5`

## API Usage

### Web Interface
1. Open http://127.0.0.1:5000
2. Go to "Change Detection" tab
3. Upload two images
4. Click "Detect Changes"

### REST API

```python
import requests

files = {
    'image1': open('before.jpg', 'rb'),
    'image2': open('after.jpg', 'rb')
}

response = requests.post('http://127.0.0.1:5000/api/change-detection', files=files)
result = response.json()

print(f"Change percentage: {result['change_percentage']}%")
```

## Requirements

- Python 3.8+
- TensorFlow 2.17+
- OpenCV
- Flask
- scikit-learn

## Model Architecture

The U-Net consists of:
- Encoder: Convolutional layers with max pooling
- Bottleneck: Deep feature extraction
- Decoder: Upsampling with skip connections
- Output: Sigmoid activation for binary change mask

## Future Improvements

- Train on real satellite image datasets (LEVIR-CD, etc.)
- Add data augmentation
- Implement post-processing (morphological operations)
- Add confidence scores
- Support for multi-temporal change detection