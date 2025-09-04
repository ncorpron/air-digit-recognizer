
import cv2
import mediapipe as mp
import numpy as np
import os

# Directory to save dataset
dataset_dir = "air_mnist_dataset/training"
os.makedirs(dataset_dir, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(dataset_dir, str(i)), exist_ok=True)

# Initialize image counters based on existing images
img_count = []
for i in range(10):
    digit_folder = os.path.join(dataset_dir, str(i))
    existing_files = [f for f in os.listdir(digit_folder) if f.endswith(".png")]
    if existing_files:
        max_index = max([int(f.split(".")[0]) for f in existing_files])
        img_count.append(max_index + 1)
    else:
        img_count.append(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

trail = []             # Stores fingertip points
canvas_size = 28       # MNIST image size
drawing = False        # Track if drawing is active
preview = False        # Track if preview is active
current_label = None   # Digit label (0-9) selected in preview

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Track fingertip if drawing is active
        if drawing and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(tip.x * w), int(tip.y * h)
            trail.append((x, y))

        # Draw trail
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 5)

        # Preview window if active
        if preview and len(trail) > 5:
            x_coords = [p[0] for p in trail]
            y_coords = [p[1] for p in trail]
            x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
            y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

            roi = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            for i in range(1, len(trail)):
                pt1 = (trail[i - 1][0] - x_min, trail[i - 1][1] - y_min)
                pt2 = (trail[i][0] - x_min, trail[i][1] - y_min)
                cv2.line(roi, pt1, pt2, 255, 10)

            roi_resized = cv2.resize(roi, (canvas_size, canvas_size))
            cv2.imshow("Preview", roi_resized)

        # Instructions
        instructions = "s:draw  v:preview  x:clear  0-9:set label  y:save  n:discard  ESC:quit"
        cv2.putText(frame, instructions, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow('Air Digit Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Start drawing
            trail = []
            drawing = True
            preview = False
        elif key == ord('v') and len(trail) > 5:  # Preview
            preview = True
            current_label = None
        elif key == ord('x'):  # Clear
            trail = []
            drawing = False
            preview = False
            current_label = None
            cv2.destroyWindow("Preview")
        elif key >= ord('0') and key <= ord('9') and preview:  # Set label
            current_label = key - ord('0')
            print(f"Selected label: {current_label}")
        elif key == ord('y') and preview and current_label is not None:  # Save
            save_path = os.path.join(dataset_dir, str(current_label),
                                     f'{img_count[current_label]}.png')
            cv2.imwrite(save_path, roi_resized)
            img_count[current_label] += 1
            print(f"Saved image: {save_path}")
            trail = []
            drawing = False
            preview = False
            current_label = None
            cv2.destroyWindow("Preview")
        elif key == ord('n') and preview:  # Discard
            print("Image discarded.")
            trail = []
            drawing = False
            preview = False
            current_label = None
            cv2.destroyWindow("Preview")

cap.release()
cv2.destroyAllWindows()
