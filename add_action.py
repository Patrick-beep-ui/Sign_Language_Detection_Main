import cv2
import numpy as np
import os
import json
import mediapipe as mp

# Load actions from JSON file and add a new action
with open('actions.json', 'r+') as f:
    data = json.load(f)
    actions = data["actions"]
    new_action = input("Enter the new action you want to add: ")
    if new_action in actions:
        print(f"The action '{new_action}' already exists.")
        exit()
    actions.append(new_action)
    data["actions"] = actions
    f.seek(0)
    json.dump(data, f, indent=4)

# Setup folder for new action data collection
DATA_PATH = os.path.join('MP_Data')
no_sequences = 30
sequence_length = 30

for sequence in range(no_sequences):
    try:
        os.makedirs(os.path.join(DATA_PATH, new_action, str(sequence)))
    except:
        pass

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def get_key_points(results):
    pose_lm = np.zeros(33*4) if results.pose_landmarks is None else np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    face_lm = np.zeros(468*3) if results.face_landmarks is None else np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    lh_lm = np.zeros(21*3) if results.left_hand_landmarks is None else np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    rh_lm = np.zeros(21*3) if results.right_hand_landmarks is None else np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([pose_lm, face_lm, lh_lm, rh_lm])

# Start video capture
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            
            if frame_num == 0:
                cv2.putText(image, f'STARTING COLLECTION for {new_action}', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {new_action}, Video {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.waitKey(1000)
            else:
                cv2.putText(image, f'Collecting frames for {new_action}, Video {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            keypoints = get_key_points(results)
            np.save(os.path.join(DATA_PATH, new_action, str(sequence), str(frame_num)), keypoints)
            cv2.imshow('Sign Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
