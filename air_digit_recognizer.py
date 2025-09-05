import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# ----- Load your trained MNIST CNN model -----
model = load_model('air_digit_cnn_trained.keras')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

trail = []             # Stores fingertip points
canvas_size = 28       # MNIST image size
drawing = False        # Flag to track if drawing is active

cap = cv2.VideoCapture(0)

# Create CLAHE object for adaptive contrast/brightness
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ----- Adjust brightness/contrast -----
        # Convert to LAB color space for better light handling
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)           # Apply CLAHE to L channel
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # ----- Track fingertip only if drawing is active -----
        if drawing and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * w), int(tip.y * h)
            trail.append((x, y))

        # ----- Draw trail -----
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 5)

        # ----- Early prediction -----
        if drawing and len(trail) > 5:
            blank = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, len(trail)):
                cv2.line(blank, trail[i - 1], trail[i], 255, 10)

            # Crop & resize
            x_coords = [p[0] for p in trail]
            y_coords = [p[1] for p in trail]
            x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
            y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

            roi = blank[y_min:y_max, x_min:x_max]
            roi = cv2.resize(roi, (canvas_size, canvas_size))
            roi = roi.astype('float32') / 255.0
            roi = roi.reshape(1, canvas_size, canvas_size, 1)

            pred = model.predict(roi)
            number = np.argmax(pred)
            confidence = np.max(pred)

            # Display early prediction
            cv2.putText(frame, f'Prediction: {number} ({confidence:.2f})',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # ----- Instructions -----
        cv2.putText(frame, "Press 's' to start, 'c' to clear, ESC to quit",
                    (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        cv2.imshow('Air Digit Recognizer', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == ord('c'):  # Clear trail and stop drawing
            trail = []
            drawing = False
        elif key == ord('s'):  # Start drawing
            trail = []
            drawing = True

cap.release()
cv2.destroyAllWindows()
