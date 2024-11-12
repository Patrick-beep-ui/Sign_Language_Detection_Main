import cv2
import numpy as np
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import time

# Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import utils as keras_utils
to_categorical = keras_utils.to_categorical

# Build and Train LSTM Neural Network
from keras import models as keras_models
from keras import layers as keras_layers
from keras import callbacks as keras_callbacks
Sequential = keras_models.Sequential
LSTM = keras_layers.LSTM
Dense = keras_layers.Dense
TensorBoard = keras_callbacks.TensorBoard


# MediaPipe values
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

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
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 250, 110), thickness=1, circle_radius=1))
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw left and right hand landmarks
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


# SETUP FOLDER FOR COLLECTION
DATA_PATH = os.path.join('MP_Data') # Path for exported data
actions = np.array(['hello', 'thanks', 'iloveyou']) # Actions to detect
no_sequences = 30 # 30 videos worth of data
sequence_length = 30 # 30 frames length for videos

for action in actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: 
            pass


# Preprocess Data and Create Labels and Features
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))  # Loop through all frames
            window.append(res)
        sequences.append(window) # 90 different videos 30 frames each
        labels.append(label_map[action])
        
x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)


# Build and Train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])  # This needs to be run to train the model

# Save the trained model
model.save('action.h5')  # Save model to H5 file

# Make Predictions
res = model.predict(x_test)
actions[np.argmax(res[0])]

# Load the model if needed later (optional)
#from keras.models import load_model
#loaded_model = load_model('action.h5')


def save_data():
    # Loop through actions
    for action in actions:
        # Loop through sequences
        for sequence in range(no_sequences):
            # Loop through sequence length
            for frame_num in range(sequence_length):
                
                # Read frame from webcam
                ret, frame = cap.read()
                
                # Perform Mediapipe detection on the frame
                image, results = mediapipe_detection(frame, holistic)
                #print(results)  # Print results for debugging
                
                # Draw landmarks on the image
                draw_landmarks(image, results)
                
                # Extract key points
                #get_key_points(results)
                
                # Wait Logic
                if frame_num == 0:
                    cv2.putText(image, 'STARING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting Frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Font Family, Font Size, Font Color, Line Width, Line Type
                    cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collecting Frames for {} Video Number {}'.format(action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Export Keypoints
                keypoints = get_key_points(results)

                                # Ensure keypoints is the expected size
                if keypoints.size == 1662:
                    keypoints = keypoints.reshape(1, 30, 1662)
                else:
                    print("Warning: Keypoints size is not as expected. Size:", keypoints.size)
                    # Handle the case when keypoints are not of expected size
                    # You could skip this iteration, use a placeholder, etc.

                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                # Flip the frame horizontally to remove mirror effect
                #flipped_frame = cv2.flip(image, 1)  # Use the processed image with the landmarks
                
                # Display the frame with landmarks
                cv2.imshow('Sign Detection', image)
                
                # Break loop if 'q' is pressed so camera is closed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

# Start video capture
cap = cv2.VideoCapture(0)
# Set and access the Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Perform Mediapipe detection on the frame
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        # Extract keypoints and reshape to fit LSTM input
        keypoints = get_key_points(results).reshape(1, 30, 1662)
        
        # Make prediction using the trained model
        prediction = model.predict(keypoints)
        action_index = np.argmax(prediction)
        action_detected = actions[action_index]
        
        # Display the detected action on the frame
        cv2.putText(image, action_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Sign Detection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    

    cap.release()
    cv2.destroyAllWindows()

