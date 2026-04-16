import requests
import os

# Test the change detection API
url = 'http://127.0.0.1:5000/api/change-detection'

# Use test images
image1_path = 'test_images/test_image.png'
image2_path = 'test_images/it logo.JPG'

if os.path.exists(image1_path) and os.path.exists(image2_path):
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {
            'image1': ('image1.png', f1, 'image/png'),
            'image2': ('image2.jpg', f2, 'image/jpeg')
        }

        response = requests.post(url, files=files)

        if response.status_code == 200:
            data = response.json()
            print("Change detection successful!")
            print(f"Change percentage: {data.get('change_percentage', 'N/A')}%")
            print(f"Algorithm: {data.get('algorithm', {}).get('method', 'Unknown')}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
else:
    print("Test images not found")