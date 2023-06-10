import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
img_size = (500, 500)
model = tf.keras.models.load_model('YOUR PATH TO YOUR TRAINED DATA')

image_number = 1
while os.path.isfile(f"test letters4/{image_number}.JPG"):
    try:
        img = cv2.imread(f"test letters4/{image_number}.JPG", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(f"Image {image_number}: predicted class is {chr(predicted_class+97)}")
        plt.imshow(img[0], cmap='gray')
        plt.title(f"Predicted Class:{chr(predicted_class+97)}")
        plt.show()
        image_number += 1
    except:
        break