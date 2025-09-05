# 🖐️ Air Digit Recognizer

![Air Digit Demo](recordings/air_digit_demo.gif)


Draw digits in the air with your hand and have them recognized by a Convolutional Neural Network (CNN) trained on MNIST!

Harness MediaPipe for hand tracking and TensorFlow/Keras for digit recognition.

---

## Features ✨
- Real-time hand tracking and fingertip detection  
- Draw digits mid-air and get instant predictions  
- Easy-to-use controls for starting, clearing, and quitting  
- Optional view and confirmation keys (v, y, n) for workflow extensions  

---

## Installation ⚡

Clone the repository:

```bash
git clone https://github.com/ncorpron/air-digit-recognizer.git
cd air-digit-recognizer
Create a virtual environment and activate it:

bash
Copy code
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage 🖱️
Run the main script:

bash
Copy code
python air_digit_recognizer.py
Controls ⌨️

Key	Action
🟢 s	Start drawing
🔴 c	Clear canvas
❌ ESC	Quit program
👁️ v	View previous predictions / toggle view
✅ y	Yes (confirm)
❌ n	No (cancel)

Model 📦
Pre-trained CNN: air_digit_cnn_trained.keras

MNIST-based digit recognition

Input size: 28×28 grayscale images

Folder Structure 🗂️
bash
Copy code
air-digit-recognizer/
│
├─ air_digit_recognizer.py      # Main hand tracking and recognition script
├─ air_digit_trainer.py          # Model training script
├─ finger_count.py               # Finger counting module
├─ hand_detection.py             # Hand detection module
├─ air_digit_cnn_trained.keras
├─ air_mnist_dataset/            # Example dataset
├─ recordings/                   # Video/GIF outputs
└─ requirements.txt
License 📜
MIT License — free to use, modify, and share.

pgsql
Copy code
