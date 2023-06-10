import tensorflow as tf
from tensorflow.keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_data_dir = "TRAINING DATA PATH"
val_data_dir = "VALIDATION DATA PATH"
img_size = (500, 500)

classes = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
    "u": 20,
    "v": 21,
    "w": 22,
    "x": 23,
    "y": 24,
    "z": 25
}

X_train = []
y_train = []

for letter in classes:
    path = os.path.join(train_data_dir, letter)
    for img_file in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        X_train.append(img)
        y_train.append(classes[letter])

X_train = np.array(X_train) / 255.0
y_train = np.array(y_train)

X_val = []
y_val = []

for letter in classes:
    path = os.path.join(val_data_dir, letter)
    for img_file in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        X_val.append(img)
        y_val.append(classes[letter])

X_val = np.array(X_val) / 255.0
y_val = np.array(y_val)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

model.save('MODEL SAVING PATH + NAME OF THE MODEL')
