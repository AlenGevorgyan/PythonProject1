import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

# --- CONFIG ---
DATASET_DIR = r'C:/Users/Admin/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train'  # <-- Update as needed
IMG_SIZE = (192, 192)
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = 'asl_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'
VALIDATION_SPLIT = 0.1

# --- DATA AUGMENTATION & GENERATORS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
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

# --- CLASS BALANCE CHECK ---
class_counts = train_generator.classes
unique, counts = np.unique(class_counts, return_counts=True)
if np.max(counts) / np.min(counts) > 2:
    print("WARNING: Your classes are imbalanced. Consider adding more images to underrepresented classes.")

# --- TRANSFER LEARNING MODEL (MobileNetV2) ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

inputs = Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# --- LOSS WITH LABEL SMOOTHING ---
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=loss_fn,
              metrics=['accuracy'])

# --- CALLBACKS ---
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# --- TRAINING ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, lr_reduce]
)

# --- UNFREEZE AND FINE-TUNE ---
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss_fn,
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    epochs=10,  # Fine-tune for a few more epochs
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint, lr_reduce]
)

# --- SAVE MODEL & CLASS INDICES ---
model.save(MODEL_PATH)
with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(train_generator.class_indices, f)
print(f"Model saved to {MODEL_PATH}")
print(f"Class indices saved to {CLASS_INDICES_PATH}")

# --- INSTRUCTIONS ---
print("\n--- INSTRUCTIONS ---")
print("1. Set DATASET_DIR to your ASL Alphabet training folder.")
print("2. Place kaggle.json in your user .kaggle folder or project root if using Kaggle download.")
print("3. Run this script: python train_model.py")
print("4. The best model will be saved as 'asl_model.h5'.")
print("5. Class indices are saved as 'class_indices.json'.") 