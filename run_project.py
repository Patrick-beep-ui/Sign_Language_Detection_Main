import cv2
import numpy as np
import os
import json
import mediapipe as mp
from keras import models as keras_models

load_model = keras_models.load_model

# Load the trained model
model = load_model('action.h5')

# Define the actions (labels) array
#actions = np.array(['hello', 'thanks', 'iloveyou'])

# Load the actions from the JSON file
with open('actions.json', 'r') as f:
    data = json.load(f)
    actions = np.array(data["actions"]) 

# MediaPipe values
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Initialize a buffer to hold the keypoints
keypoint_buffer = []
buffer_size = 30  # Size of the buffer for frames

# Detect with Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
    image.flags.writeable = False  # Prevent modifications during processing
    results = model.process(image)  # Perform prediction
    image.flags.writeable = True  # Allow modifications again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    return image, results

# Draw landmarks on image
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Extract Key Points
def get_key_points(results):
    pose_lm = np.zeros(33*4)  # Default to zeros
    face_lm = np.zeros(468*3)  # Default to zeros
    lh_lm = np.zeros(21*3)     # Default to zeros
    rh_lm = np.zeros(21*3)     # Default to zeros
    
    if results.pose_landmarks:
        pose_lm = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        
    if results.face_landmarks:
        face_lm = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()

    if results.left_hand_landmarks:
        lh_lm = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        
    if results.right_hand_landmarks:
        rh_lm = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose_lm, face_lm, lh_lm, rh_lm])

# Start video capture
cap = cv2.VideoCapture(0)

# Set and access the Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Perform Mediapipe detection on the frame
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        # Extract keypoints
        keypoints = get_key_points(results) 
        
        # Debugging: print the size and contents of keypoints
        #print("Keypoints shape:", keypoints.shape)
        #print("Keypoints contents:", keypoints)

        # Append keypoints to the buffer
        keypoint_buffer.append(keypoints)

        # If we have enough keypoints (buffer filled), make a prediction
        if len(keypoint_buffer) == buffer_size:
            # Convert the list to a numpy array and reshape to (1, 30, 1662)
            prediction_input = np.array(keypoint_buffer).reshape(1, buffer_size, 1662)
            prediction = model.predict(prediction_input)  # Predict
            action_index = np.argmax(prediction)
            action_detected = actions[action_index]
            
            # Clear the buffer for the next set of predictions
            keypoint_buffer = []

            # Display the detected action on the frame
            cv2.putText(image, action_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            print("Waiting for more keypoints...")

        # Show the frame
        cv2.imshow('Sign Detection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
