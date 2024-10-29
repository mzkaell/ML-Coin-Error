# ML-Coin-Error
ML model to detect coin erorrs
Overview
This project focuses on training a deep learning model to classify images of double die coins. The dataset consists of images and corresponding XML files containing labels for the classification task. The model is built using TensorFlow and Keras.

Requirements
Python 3.x
TensorFlow
Keras
pandas
scikit-learn
Google Colab (for mounting Google Drive and accessing datasets)
Dataset Structure
The dataset is organized into two main folders:

Train/: Contains training images and their corresponding XML files.
Val/: Contains validation images and their corresponding XML files.
Each image file has an associated XML file that provides the label for the image.

Instructions
Mount Google Drive: Ensure you have your dataset zipped and uploaded to Google Drive. Update the zip_file_path variable with the correct path to your dataset zip file.

Unzip the Dataset: The script will automatically unzip the dataset into a specified directory.

Load Data: The code loads images and their labels from the specified training and validation directories.

Data Preprocessing: Images are preprocessed using ImageDataGenerator to augment the training set, including operations like rotation, shifting, shearing, and flipping.

Model Definition: A convolutional neural network (CNN) is defined with several convolutional and pooling layers, followed by dense layers for classification.

Training: The model is trained on the training dataset with validation on the validation dataset, utilizing early stopping to prevent overfitting.

Evaluation: Finally, the model evaluates its performance on the validation set, providing accuracy metrics.
