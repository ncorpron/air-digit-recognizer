import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "air_mnist_dataset/training"
model_path = "air_digit_cnn_trained.keras"
canvas_size = 28

# Load dataset
X, y = [], []
for digit in range(10):
    digit_dir = os.path.join(dataset_dir, str(digit))
    for file in os.listdir(digit_dir):
        if file.endswith(".png"):
            path = os.path.join(digit_dir, file)
            img = load_img(path, color_mode="grayscale", target_size=(canvas_size, canvas_size))
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(digit)

X = np.array(X, dtype="float32") / 255.0
y = to_categorical(y, num_classes=10)

# Add channel dim
if X.shape[-1] != 1:
    X = X.reshape(-1, canvas_size, canvas_size, 1)

# Split train/val
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = load_model(model_path)

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

