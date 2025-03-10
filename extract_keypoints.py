import os
import numpy as np
import cv2
import mediapipe as mp

# Initialize Mediapipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create empty lists for keypoints and labels
X, y = [], []

# Path to dataset
data_dir = "data/"

# Process dataset
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    
    if not os.path.isdir(label_dir):
        continue

    print(f"Processing pose: {label}")

    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Skipping {img_path} (Not a valid image)")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            X.append(np.array(keypoints).flatten())  # Save keypoints
            y.append(label)  # Save corresponding label
        else:
            print(f"Skipping {img_path} (No pose detected)")

# Convert lists to arrays and save
np.save("X.npy", np.array(X))
np.save("y.npy", np.array(y))

print("✅ Keypoint extraction complete. Data saved as X.npy and y.npy!")