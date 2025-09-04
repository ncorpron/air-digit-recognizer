import cv2
import mediapipe as mp

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break
cap.release()
cv2.destroyAllWindows()
