import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model('sign_language_model.h5')
classes = np.load('classes.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
img_size = 64

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Bounding box
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmark.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmark.landmark]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            # Add some padding
            x1 = max(x1 - 20, 0)
            y1 = max(y1 - 20, 0)
            x2 = min(x2 + 20, w)
            y2 = min(y2 + 20, h)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (img_size, img_size))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)
            class_id = np.argmax(prediction)
            class_label = classes[class_id]

            cv2.putText(frame, class_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

