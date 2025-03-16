# Yoga Pose Detection Web Application

### By: Peethala V Siva Sampath Likhit  

This project involves developing a web-based application for **Yoga Pose Detection**. The goal of this application is to use deep learning models to identify various yoga poses from an uploaded image. The app also displays reference images for different yoga poses and provides a user-friendly interface built with **Streamlit**.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
  - [Install Dependencies](#install-dependencies)
  - [Running the Application](#running-the-application)
- [Model Architecture](#model-architecture)
- [How it Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Introduction

The **Yoga Pose Detection Web App** uses a deep learning model to classify images into different yoga poses. Users can upload an image, and the model will predict the yoga pose from the image. Additionally, the app provides reference images for different yoga poses, allowing users to compare their poses with reference images.

---

## Features

- **Pose Prediction**: Upload an image to detect yoga poses.
- **Pose Reference Images**: View reference images of popular yoga poses.
- **Model Accuracy**: Displays the model's accuracy after predictions.
- **User Interface**: Developed using **Streamlit**, a fast way to create web applications for data science projects.

---

## Technologies Used

- **Python**: The main programming language used for the project.
- **TensorFlow/Keras**: Used for training the deep learning model.
- **Streamlit**: Framework for building the web application.
- **OpenCV**: Used for reading images and performing image processing tasks.
- **MediaPipe**: Used for extracting keypoints from images for pose detection.
- **NumPy**: For handling array operations and numerical data.
- **scikit-learn**: For preprocessing and splitting the dataset.

---

## Setup Instructions

### Install Dependencies

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/LikhitP12/yoga-pose-detection.git
    cd yoga-pose-detection
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On **Windows**:
      ```bash
      venv\Scripts\activate
      ```
    - On **macOS/Linux**:
      ```bash
      source venv/bin/activate
      ```

4. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To start the application locally, run the following command in your terminal:
```bash
streamlit run app.py
This will launch the application in your default web browser, where you can interact with the Yoga Pose Detection Web App.

---

## Model Architecture

The model used for yoga pose detection is a **Convolutional Neural Network (CNN)**, trained on keypoint-based features extracted using **MediaPipe**. The architecture of the model consists of several convolutional layers followed by dense layers for classification.

Key highlights:

* **Preprocessing**: Keypoints are extracted from images using MediaPipe.
* **Model**: A deep neural network consisting of multiple dense layers.
* **Output**: The model outputs the probability distribution over several yoga poses.

---

## How it Works

1. **Pose Detection**: The user uploads an image of a yoga pose. The application uses **MediaPipe** to extract keypoints from the image.
2. **Prediction**: The extracted keypoints are passed to the trained model, which predicts the yoga pose.
3. **Display**: The predicted pose is displayed, and reference images for the selected pose are shown.

---

## Folder Structure

```plaintext
yoga-pose-detection/
│
├── app.py                # Main Streamlit application
├── model/                # Contains trained models and label encoder
│   ├── yoga_pose_model_keypoints.h5  # Trained Keras model
│   └── label_classes.npy  # Label encoder classes
│
├── reference_images/     # Folder for storing reference images
│   ├── downdog1.jpg
│   ├── goddess1.jpg
│   └── ...
│
├── requirements.txt      # Required Python packages
