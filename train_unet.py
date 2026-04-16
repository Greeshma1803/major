"""
U-Net Training Script for Change Detection

This script demonstrates how to train the U-Net model for change detection.
For real-world use, you would need:
1. Pairs of before/after images
2. Ground truth change masks (binary masks indicating changed areas)

The training data should be in the format:
- X: concatenated before and after images (shape: H, W, 6)
- y: binary change mask (shape: H, W, 1)
"""

import numpy as np
import tensorflow as tf
from app import build_unet, preprocess_for_unet
import cv2
import os

def create_synthetic_training_data(num_samples=100, img_size=(256, 256)):
    """
    Create synthetic training data for demonstration.
    In practice, use real satellite image pairs with ground truth change masks.
    """
    X = []
    y = []

    for i in range(num_samples):
        # Create synthetic "before" image
        img1 = np.random.rand(*img_size, 3).astype(np.float32)

        # Create synthetic "after" image with some changes
        img2 = img1.copy()

        # Add some random changes (rectangles, circles, etc.)
        change_mask = np.zeros((*img_size, 1), dtype=np.float32)

        # Add random rectangles as changes
        for _ in range(np.random.randint(1, 5)):
            x1 = np.random.randint(0, img_size[1]-50)
            y1 = np.random.randint(0, img_size[0]-50)
            x2 = x1 + np.random.randint(20, 50)
            y2 = y1 + np.random.randint(20, 50)

            # Modify the "after" image
            img2[y1:y2, x1:x2] = np.random.rand(y2-y1, x2-x1, 3)

            # Mark as changed in ground truth
            change_mask[y1:y2, x1:x2] = 1.0

        # Concatenate images
        combined = np.concatenate([img1, img2], axis=-1)
        X.append(combined)
        y.append(change_mask)

    return np.array(X), np.array(y)

def train_unet_model():
    """Train the U-Net model for change detection"""

    print("Building U-Net model...")
    model = build_unet()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("Creating synthetic training data...")
    X_train, y_train = create_synthetic_training_data(num_samples=200)

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )

    # Save the trained model
    model.save('unet_change_detection.h5')
    print("Model saved as 'unet_change_detection.h5'")

    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_unet_model()

    # Test the model
    print("\nTesting trained model...")
    X_test, y_test = create_synthetic_training_data(num_samples=10)

    predictions = model.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(np.float32)

    # Calculate some metrics
    intersection = np.sum(predictions_binary * y_test)
    union = np.sum(predictions_binary) + np.sum(y_test) - intersection
    iou = intersection / union if union > 0 else 0

    print(".3f")
    print(".3f")