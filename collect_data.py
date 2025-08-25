import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

labels = ["A", "B", "C", "D", "E"]
output_folder = "dataset"

for label in labels:
    os.makedirs(f"{output_folder}/{label}", exist_ok=True)

cap = cv2.VideoCapture(0)
current_label = "A"
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in handLms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            np.save(f"{output_folder}/{current_label}/{count}.npy", np.array(landmarks))
            count += 1

    cv2.putText(frame, f"Collecting: {current_label} ({count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1)
    if key == ord('n'):
        current_label = input("Enter next label: ").upper()
        count = 0
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
