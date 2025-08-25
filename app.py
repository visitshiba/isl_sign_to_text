import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated.*")

import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and label map
model, label_map = joblib.load("gesture_classifier.pkl")  # Make sure this file exists
reverse_map = {v: k for k, v in label_map.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract normalized landmark coordinates (x and y)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Debug: check landmarks shape and values
            print(f"Landmarks extracted: {landmarks}")

            # Predict the sign from landmarks
            try:
                pred = model.predict([landmarks])[0]
                prediction_text = reverse_map.get(pred, "Undefined")
                print(f"Prediction: {prediction_text}")
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction_text = "Undefined"

    # Display the prediction on the frame
    cv2.putText(frame, f'Sign: {prediction_text}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ISL Sign to Text", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
