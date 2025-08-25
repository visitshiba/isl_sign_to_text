import cv2
import mediapipe as mp
import numpy as np
import os
import json

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Folder to save collected data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load existing dataset or create new
data_file = os.path.join(DATA_DIR, "data.json")
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        dataset = json.load(f)
else:
    dataset = {"landmarks": [], "labels": []}

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")
print("Press 's' to save current frame's landmarks with label.")

label = input("Enter label for the gesture you want to collect data for: ").strip()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            cv2.putText(frame, "Press 's' to save this gesture", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Show your hand clearly", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Collect ISL Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if results.multi_hand_landmarks:
            dataset["landmarks"].append(landmarks)
            dataset["labels"].append(label)
            print(f"Saved data point for label '{label}'. Total samples: {len(dataset['labels'])}")
        else:
            print("No hand detected, cannot save")

# Save collected data to disk
with open(data_file, "w") as f:
    json.dump(dataset, f)

print(f"Data saved to {data_file}")

cap.release()
cv2.destroyAllWindows()
