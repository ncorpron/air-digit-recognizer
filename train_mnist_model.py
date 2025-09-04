# train_mnist_model.py
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----- Paths & parameters -----
data_dir = 'data'  # folder containing subfolders 0,1,...,9 with your images
img_size = (28, 28)
batch_size = 32
epochs = 10  # adjust as needed

# ----- Image generators -----
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation',
    shuffle=True
)

# ----- Build CNN model -----
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----- Train model -----
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# ----- Save model -----
model.save('mnist_cnn_updated.keras')
print("Updated model saved as mnist_cnn_updated.keras")
