# -*- coding: utf-8 -*-
"""Coin Error Image Analysis"""

import os
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile
from google.colab import files
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

from google.colab import drive
drive.mount('/content/drive')

zip_file_path = '/content/drive/My Drive/Double Die.zip'

# Create a directory for the unzipped contents if it doesn't exist
unzip_dir = '/content/Double Die'
os.makedirs(unzip_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# List the contents of the unzipped directory for debugging
print(os.listdir(unzip_dir))

# Set the data folder where your images and XML files are located
train_folder = '/content/Double Die/Train/'
val_folder = '/content/Double Die/Val/'

print(os.listdir(train_folder))
print(os.listdir(val_folder))

# Function to load images and labels from a specified folder
def load_data(folder):
    images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    labels = []

    for img in images:
        xml_file = img.replace('.jpg', '.xml')
        xml_path = os.path.join(folder, xml_file)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the label (adjust based on your XML structure)
            for obj in root.findall('object'):
                label = obj.find('name').text
                label = label.lower().replace(" ", "")
                labels.append(label)
                print(f'Image: {img}, Label: {label}')  # Debugging output
        else:
            print(f'Missing XML for {img}')
            labels.append('singledie')  # Default label for missing XML files

    return images, labels

train_images, train_labels = load_data(train_folder)
train_data = {'filename': train_images, 'class': train_labels}
train_df = pd.DataFrame(train_data)

# Load validation data
val_images, val_labels = load_data(val_folder)
val_data = {'filename': val_images, 'class': val_labels}
val_df = pd.DataFrame(val_data)

print(train_df.shape)
print(val_df.shape)

# Combine and split to ensure stratification
if len(train_df) > 0 and len(val_df) > 0:
  train_df, val_df = train_test_split(
      pd.concat([train_df, val_df]),  # Combine and then split
      test_size=0.2,  # Adjust the test_size as needed
      random_state=42,  # Set a random seed for reproducibility
      stratify=pd.concat([train_df, val_df])['class']  # Ensure class distribution
  )
elif len(train_df) == 0:
  print ("Error: train_df is empty")
elif len(val_df) == 0:
  print ("Error: val_df is empty")
  train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['class'])

def validate_filenames(df, folder):
    valid_files = [filename for filename in df['filename'] if os.path.exists(os.path.join(folder, filename))]
    return valid_files

train_df['valid'] = train_df['filename'].apply(lambda x: os.path.exists(os.path.join(train_folder, x)))
val_df['valid'] = val_df['filename'].apply(lambda x: os.path.exists(os.path.join(val_folder, x)))

print("Valid training filenames:", train_df[train_df['valid']].shape[0])
print("Valid validation filenames:", val_df[val_df['valid']].shape[0])

train_df = train_df[train_df['valid']].drop(columns=['valid'])
val_df = val_df[val_df['valid']].drop(columns=['valid'])

# Initialize ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

label_encoder = LabelEncoder()

# Fit the encoder on all unique labels from both train and validation sets
all_labels = pd.concat([train_df['class'], val_df['class']]).unique()
label_encoder.fit(all_labels)

# Create flow generators from the DataFrames
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_folder,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_folder,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)

for batch in train_generator:
    x, y = batch
    print(f'Batch shape: {x.shape}, Labels: {y}')
    break

# === TRANSFER LEARNING MODEL SETUP ===
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Load MobileNetV2 base model with pretrained ImageNet weights, exclude top layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze base model

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights to handle imbalance
train_labels_encoded = label_encoder.transform(train_df['class'])
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Train the model
if train_df.empty or val_df.empty:
    print("Error: train_df or val_df is empty.")
else:
  history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=2,
    epochs=10,
    callbacks=[early_stopping],
    class_weight=class_weights_dict
)

# Optional: Fine-tuning for better accuracy (uncomment to run)
'''
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower lr for fine tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=2,
    epochs=5,
    callbacks=[early_stopping],
    class_weight=class_weights_dict
)
'''

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
