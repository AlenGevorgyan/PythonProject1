import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import json

# --- CONFIG ---
MODEL_PATH = 'asl_model.h5'
TESTSET_DIR = r'C:\Users\Admin\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_test\asl_alphabet_test'  # <-- This should be the folder containing your test images or class subfolders
IMG_SIZE = (200, 200)  # FIXED: Match training input size
BATCH_SIZE = 32

# Load model
model = load_model(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")

# Load class indices (if available)
if os.path.exists('class_indices.json'):
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    # Fallback: use default ASL letters
    import string
    idx_to_class = {i: l for i, l in enumerate(list(string.ascii_uppercase) + ["del", "nothing", "space"])}

# List all image files in the test directory
image_files = [f for f in os.listdir(TESTSET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

correct = 0
total = 0

for img_name in image_files:
    img_path = os.path.join(TESTSET_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_name}")
        continue
    
    # FIXED: Proper preprocessing to match training
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_resized = img_resized / 255.0
    img_resized = img_resized.astype('float32')
    img_resized = np.expand_dims(img_resized, axis=0)
    
    pred = model.predict(img_resized, verbose=0)
    pred_idx = np.argmax(pred)
    pred_label = idx_to_class[pred_idx]
    confidence = np.max(pred)

    # Try to extract the true label from the filename (e.g., 'A_test.jpgй' -> 'A')
    true_label = img_name[0].upper()
    print(f"Image: {img_name} | Predicted: {pred_label} (conf: {confidence:.3f}) | True: {true_label}")

    if pred_label.upper() == true_label:
        correct += 1
    total += 1

if total > 0:
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
else:
    print("No images found for testing.")

print("\n--- INSTRUCTIONS ---")
print("1. Place your test images in the folder specified by TESTSET_DIR.")
print("2. Each image filename should start with the true label (e.g., 'A_test.jpg').")
print("3. Run this script: python test_single_images.py")