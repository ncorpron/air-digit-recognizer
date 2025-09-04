import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    fingers = []

    # ---- Thumb: dynamic distance check ----
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Hand size reference: wrist to middle finger MCP
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    hand_size = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)

    # Thumb extension distance
    thumb_extension = math.sqrt((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)

    if thumb_extension > 0.5 * hand_size:  # only count if thumb extended
        fingers.append(1)
    else:
        fingers.append(0)

    # ---- Other 4 fingers: tip above PIP and MCP ----
    for tip_id in [8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        mcp = hand_landmarks.landmark[tip_id - 3]

        fingers.append(1 if tip.y < pip.y and tip.y < mcp.y else 0)

    return fingers

# ----- Main -----
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        total_fingers = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                fingers = count_fingers(hand_landmarks)
                count = sum(fingers)
                total_fingers += count

                # Label each hand
                hand_label = handedness.classification[0].label
                coords = hand_landmarks.landmark[0]  # wrist for text position
                h, w, _ = frame.shape
                x, y = int(coords.x * w), int(coords.y * h)
                cv2.putText(frame, f"{hand_label}: {count}", (x - 30, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Total fingers
        cv2.putText(frame, f'Total: {total_fingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('Finger Counter', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
