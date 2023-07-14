![image](https://github.com/ybryan95/CIFAR_DBN_Demo/assets/123009743/ce70408c-b1ac-4902-b453-bc28e4853e78)

# CIFAR-10 Image Classification with Convolutional Neural Networks (CNN)

This project provides a simple demonstration of how to use a Convolutional Neural Network (CNN) to perform image classification on the CIFAR-10 dataset. This dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

# Table of Contents
- [1. Getting started](#1-getting-started)
    - [1.1 Dataset](#11-dataset)
    - [1.2 Prerequisites](#12-prerequisites)
- [2. The Code Explained](#2-the-code-explained)
    - [2.1 Data Loading and Preprocessing](#21-data-loading-and-preprocessing)
    - [2.2 Building the CNN Model](#22-building-the-cnn-model)
    - [2.3 Training the Model](#23-training-the-model)
    - [2.4 Evaluation and Visualization](#24-evaluation-and-visualization)
- [3. Contributing](#3-contributing)


# 1. Getting started
## 1.1 Dataset
The dataset used is the CIFAR-10 dataset, a well-established dataset for image classification, which is readily available in Keras.

## 1.2 Prerequisites
This project is implemented in Python using Keras for building the CNN. Ensure that you have Keras installed, along with dependencies like NumPy and Matplotlib for handling data and visualizing model performance.

# 2. The Code Explained
## 2.1 Data Loading and Preprocessing
The data is loaded from Keras and then normalized. Each pixel in the image data is divided by 255.0 to normalize it to the range 0-1. The labels are one-hot encoded to represent the 10 different classes of images.

## 2.2 Building the CNN Model
The CNN is built as a sequential model with two sets of Convolutional and MaxPooling layers, each followed by a Flatten layer and then two Dense layers. The first Dense layer uses ReLU activation and the last Dense layer (which is the output layer) uses Softmax activation for multi-class classification.

## 2.3 Training the Model
The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and it uses accuracy as the evaluation metric. The model is then trained on the training data for 20 epochs with a batch size of 128.

## 2.4 Evaluation and Visualization
The model is evaluated on the test data, providing the loss and accuracy. These results are then plotted over the training period to visualize the performance of the model.

# 3. Contributing
Feel free to submit pull requests or propose changes. For major changes, please open an issue first to discuss the change you wish to make.

