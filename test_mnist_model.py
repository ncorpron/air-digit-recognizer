import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = to_categorical(y_test, 10)

# Load your saved model
model = load_model('mnist_cnn.keras')

# Make predictions on first 5 test images
predictions = model.predict(x_test[:5])
print("Predicted digits:", predictions.argmax(axis=1))
print("True digits:     ", y_test[:5].argmax(axis=1))
