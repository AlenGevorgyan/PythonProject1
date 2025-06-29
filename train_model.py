import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# --- CONFIG ---
DATASET_DIR = r'C:\Users\Admin\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train'  # <-- This should be the folder containing all class subfolders
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = 'asl_model.h5'
CONTINUE_TRAINING = False

# --- DATA AUGMENTATION & GENERATORS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- MODEL DEFINITION OR LOADING ---
if CONTINUE_TRAINING and os.path.exists(MODEL_PATH):
    print("Loading existing model for further training...")
    model = load_model(MODEL_PATH)
else:
    print("Building a new model...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- CALLBACKS ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

# --- TRAINING ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)

# --- SAVE MODEL (redundant if checkpoint used, but ensures latest is saved) ---
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- SAVE CLASS INDICES ---
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices saved to class_indices.json")

# --- INSTRUCTIONS ---
print("\n--- INSTRUCTIONS ---")
print("1. Set DATASET_DIR to your ASL Alphabet training folder.")
print("2. Set CONTINUE_TRAINING = True if you want to continue training an existing model.")
print("3. Run this script: python train_model.py")
print("4. The best model will be saved as 'asl_model.h5'.")
print("5. Class indices are saved as 'class_indices.json'.") 