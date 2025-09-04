import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Parameters
dataset_dir = "air_mnist_dataset/training"
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

# Add channel dimension if needed
if X.shape[-1] != 1:
    X = X.reshape(-1, canvas_size, canvas_size, 1)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(canvas_size, canvas_size, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

# Evaluate on validation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Save model
model.save("air_digit_cnn_trained.keras")
print("Model saved as air_digit_cnn_trained.keras")

