import cv2
import numpy as np
import json
import joblib
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

MODEL_PATH = 'Model_Output/action_best_model.h5' 
SCALER_PATH = 'Model_Output/scaler.save'
LABELS_PATH = 'Model_Output/actions.json'

SEQUENCE_LENGTH = 30
PREDICTION_SMOOTHING = 5
CONFIDENCE_THRESHOLD = 0.75
MOUSE_SMOOTHING = 5
FRAME_MARGIN = 100
DEADZONE_RADIUS = 3

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

try:
    print("Dang tai tai nguyen...")
    with open(LABELS_PATH, 'r') as f:
        actions = np.array(json.load(f))
    
    scaler = joblib.load(SCALER_PATH)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 63)))
    model.add(Dropout(0.4))
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(actions), activation='softmax'))

    print("Dang nap trong so Model...")
    model.load_weights(MODEL_PATH)
    print("San sang! Hay dua tay len truoc camera.")

except Exception as e:
    print(f"Loi khoi tao: {e}")
    print("Hay dam bao da cai tensorflow==2.15.0")
    exit()

def get_relative_keypoints(frame_data):
    landmarks = frame_data.reshape(21, 3)
    base_x, base_y, base_z = landmarks[0][0], landmarks[0][1], landmarks[0][2]
    landmarks[:, 0] -= base_x
    landmarks[:, 1] -= base_y
    landmarks[:, 2] -= base_z
    return landmarks.flatten()

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sequence = []
predictions = deque(maxlen=PREDICTION_SMOOTHING)
plocX, plocY = 0, 0
clocX, clocY = 0, 0
prev_action = None
missed_frames = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN), (w - FRAME_MARGIN, h - FRAME_MARGIN), (255, 0, 255), 2)

        if results.multi_hand_landmarks:
            missed_frames = 0
            hand_lms = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            lm_list = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            processed_kp = get_relative_keypoints(lm_list.copy())
            
            sequence.append(processed_kp)
            sequence = sequence[-SEQUENCE_LENGTH:]
            
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0).reshape(-1, 63)
                input_scaled = scaler.transform(input_data).reshape(1, SEQUENCE_LENGTH, 63)
                
                res = model.predict(input_scaled, verbose=0)[0]
                predictions.append(res)
                
                avg_res = np.mean(predictions, axis=0)
                best_idx = np.argmax(avg_res)
                confidence = avg_res[best_idx]
                current_action = actions[best_idx]
                
                cv2.putText(frame, f"{current_action} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if confidence > CONFIDENCE_THRESHOLD:
                    if current_action == 'Move':
                        prev_action = 'Move'
                        x1 = hand_lms.landmark[8].x * w
                        y1 = hand_lms.landmark[8].y * h
                        target_x = np.interp(x1, (FRAME_MARGIN, w - FRAME_MARGIN), (0, screen_w))
                        target_y = np.interp(y1, (FRAME_MARGIN, h - FRAME_MARGIN), (0, screen_h))
                        clocX = plocX + (target_x - plocX) / MOUSE_SMOOTHING
                        clocY = plocY + (target_y - plocY) / MOUSE_SMOOTHING
                        
                        if math.hypot(clocX - plocX, clocY - plocY) > DEADZONE_RADIUS: 
                             pyautogui.moveTo(clocX, clocY)
                             plocX, plocY = clocX, clocY
                    
                    elif current_action == 'ScrollUp': pyautogui.scroll(50) 
                    elif current_action == 'ScrollDown': pyautogui.scroll(-50)
                    else:
                        if current_action != prev_action:
                            if current_action == 'LeftClick': pyautogui.click()
                            elif current_action == 'RightClick': pyautogui.rightClick()
                            elif current_action == 'ZoomIn': pyautogui.hotkey('ctrl', '+')
                            elif current_action == 'ZoomOut': pyautogui.hotkey('ctrl', '-')
                            elif current_action == 'NextSlide': pyautogui.press('right')
                            elif current_action == 'PrevSlide': pyautogui.press('left')
                            prev_action = current_action
        else:
            missed_frames += 1
            if missed_frames > 15: 
                sequence = []
                predictions.clear()
                prev_action = None
                cv2.putText(frame, "No Hand Detected", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Smart Hand Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()