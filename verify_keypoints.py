import numpy as np

# Load the extracted keypoints and labels
try:
    X = np.load("X.npy")
    y = np.load("y.npy")
except FileNotFoundError:
    print("❌ X.npy or y.npy not found. Make sure you've run extract_keypoints.py!")
    exit()

# Display dataset statistics
print("✅ Keypoint Extraction Summary")
print("Shape of X (features):", X.shape)  # Expected: (number of images, 99)
print("Shape of y (labels):", y.shape)  # Expected: (number of images,)
print("Unique Labels:", np.unique(y))  # Expected: ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
print("First 5 Labels:", y[:5])
