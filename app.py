import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp
import os

# ✅ Load the keypoint-based model
model = keras.models.load_model("yoga_pose_model_keypoints.h5")

# ✅ Load label encoder mappings
label_encoder = np.load("label_classes.npy", allow_pickle=True)

# ✅ Yoga Poses and Reference Images
labels = {
    "Downdog": ["downdog1.jpg", "downdog2.jpg", "downdog3.jpg"],
    "Goddess": ["goddess1.jpg", "goddess2.jpg", "goddess3.jpg"],
    "Plank": ["plank1.jpg", "plank2.jpg", "plank3.jpg"],
    "Tree": ["tree1.jpg", "tree2.jpg", "tree3.jpg"],
    "Warrior2": ["warrior2_1.jpg", "warrior2_2.jpg", "warrior2_3.jpg"]
}

# ✅ Model accuracy (Update after training)
MODEL_ACCURACY = 90.0  # Change this after evaluating

# ✅ Initialize Mediapipe for Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ✅ Function to extract keypoints from an image
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
        return np.array(keypoints).reshape(1, -1)
    return None

# ✅ Set Page Title and Icon
st.set_page_config(page_title="Yoga Pose Detection", page_icon="🧘", layout="wide")

# ✅ Sidebar with Navigation
st.sidebar.title("🧘 Yoga Pose Detection")
st.sidebar.markdown("#### **By: Peethala V Siva Sampath Likhit**")
st.sidebar.markdown("#### **Roll No: 2024028407**")
st.sidebar.markdown("#### **MTech CSE**")
st.sidebar.markdown("---")

# ✅ Sidebar Navigation
menu_option = st.sidebar.radio(
    "📌 Select an Option",
    ["🔍 Detect Yoga Pose", "📸 View Yoga Pose References"]
)

# ✅ Main Header
st.markdown(
    """
    <h1 style='text-align: center; color: #3498db;'>🧘‍♂️ Yoga Pose Detection Web App</h1>
    <h4 style='text-align: center; color: #2c3e50;'>Upload an image and let the model predict the yoga pose!</h4>
    """, unsafe_allow_html=True
)

# ✅ Display Model Accuracy
st.success(f"🎯 **Model Accuracy:** {MODEL_ACCURACY}%")

# ✅ Menu Option: Detect Yoga Pose
if menu_option == "🔍 Detect Yoga Pose":
    uploaded_file = st.file_uploader("📸 **Upload a yoga pose image...**", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Display Uploaded Image
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Extract keypoints
        keypoints = extract_keypoints(image)

        if keypoints is not None:
            # Predict the pose
            prediction = model.predict(keypoints)
            predicted_label = label_encoder[np.argmax(prediction)]

            # Display Prediction Result
            st.markdown(
                f"<h3 style='text-align: center; color: #27ae60;'>✅ Predicted Pose: {predicted_label}</h3>",
                unsafe_allow_html=True
            )
        else:
            st.error("❌ No pose detected. Please try another image.")

# ✅ Menu Option: View Reference Yoga Poses
elif menu_option == "📸 View Yoga Pose References":
    pose_selected = st.selectbox("📌 **Choose a Yoga Pose:**", list(labels.keys()))

    if pose_selected:
        st.markdown(f"### {pose_selected} Pose Reference Images")
        cols = st.columns(3)
        for idx, img_file in enumerate(labels[pose_selected]):
            img_path = os.path.join("reference_images", img_file)
            with cols[idx]:
                st.image(img_path, caption=f"{pose_selected} Pose {idx+1}", use_column_width=True)

# ✅ Footer
st.markdown(
    """
    <hr>
    <h5 style='text-align: center; color: #34495e;'>Developed by Peethala V Siva Sampath Likhit | Roll No: 2024028407 | MTech CSE</h5>
    """, unsafe_allow_html=True
)