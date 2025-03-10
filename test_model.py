import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("yoga_pose_model.h5")

# Load label mappings
labels = np.load("label_classes.npy")

# Generate a random test input (simulate a real pose)
random_pose = np.random.rand(1, 99)

# Predict the pose
prediction = model.predict(random_pose)
predicted_label = labels[np.argmax(prediction)]

print(f"✅ Model is working! Predicted Pose: {predicted_label}")