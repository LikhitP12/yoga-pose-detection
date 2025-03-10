import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Load keypoints dataset (X: features, y: labels)
X = np.load("X.npy")
y = np.load("y.npy")

# ✅ Convert string labels to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e.g., 'downdog' → 0, 'plank' → 1, etc.
np.save("label_classes.npy", label_encoder.classes_)  # Save class mappings

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ Convert labels to categorical (One-hot encoding)
num_classes = len(label_encoder.classes_)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# ✅ Define a simple Deep Neural Network (DNN) for keypoints
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')  # Output layer
])

# ✅ Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train Model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# ✅ Save Model
model.save("yoga_pose_model_keypoints.h5")
print("✅ Model training complete. Model saved as yoga_pose_model_keypoints.h5!")
