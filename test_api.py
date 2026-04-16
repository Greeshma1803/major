import requests
import base64
import os

# Test health
print("=== Health Check ===")
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Test prediction
print("\n=== Single Prediction ===")
image_path = r'C:\Users\ADMIN\landuse_infer\test_images\test_image.png'

with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(f"Predicted: {result['data']['class']} ({result['data']['confidence']}%)")

# Test annotation
print("\n=== Image Annotation ===")
with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/annotate', files=files)
    result = response.json()
    
    # Save annotated image
    img_data = base64.b64decode(result['data']['annotated_image'])
    with open('annotated_result.png', 'wb') as img_file:
        img_file.write(img_data)
    print(f"Annotated image saved as 'annotated_result.png'")
    print(f"Top prediction: {result['data']['top_prediction']['class']}")