import cv2
import numpy as np
import os
import json
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import utils as keras_utils
from keras import models as keras_models
from keras import layers as keras_layers

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
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Extract Key Points
def get_key_points(results):
    pose_lm = np.zeros(33 * 4)
    face_lm = np.zeros(468 * 3)
    lh_lm = np.zeros(21 * 3)
    rh_lm = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose_lm = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    if results.face_landmarks:
        face_lm = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    if results.left_hand_landmarks:
        lh_lm = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh_lm = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose_lm, face_lm, lh_lm, rh_lm]) 

# Set up folder for collection
DATA_PATH = os.path.join('MP_Data') 
#actions = np.array(['hello', 'thanks', 'iloveyou']) 
no_sequences = 30 
sequence_length = 30 

with open('actions.json', 'r') as f:
    data = json.load(f)
    actions = np.array(data["actions"]) 

# Preprocess data and create labels and features
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))  
            window.append(res)
        sequences.append(window) 
        labels.append(label_map[action])
        
x = np.array(sequences)
y = keras_utils.to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

# Build and train LSTM Neural Network
model = keras_models.Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000)

# Save the trained model
model.save('action.h5')
