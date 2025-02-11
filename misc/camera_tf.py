# importing the modules
import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import resize

import config

frameWidth = 640
frameHeight = 360

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.resize_tensor_input(
    interpreter.get_input_details()[0]["index"], [1, 3, frameHeight, frameWidth]
)
interpreter.allocate_tensors()

output_details = interpreter.get_output_details()
input_name = interpreter.get_input_details()[0]["index"]
output_name = interpreter.get_output_details()[0]["index"]

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgResult = img.copy()
    img2 = cv2.resize(imgResult, (config.REFINER_SIZE, config.REFINER_SIZE))
    img2 = (img2[np.newaxis, ...].transpose(0, 3, 1, 2) / 255).astype(np.float32)

    interpreter.set_tensor(input_name, img2)
    interpreter.invoke()
    out = interpreter.get_tensor(output_name)[0][1]
    out = resize(out, (frameHeight, frameWidth))
    out[out > 0.7] = 1
    out[out <= 0.4] = 0
    out = np.stack([out, out, out], axis=2)
    out = cv2.GaussianBlur(out, (15, 15), 0)
    blurred = cv2.GaussianBlur(imgResult, (101, 101), 0)
    imgResult = (imgResult * out + blurred * (1 - out)).astype(np.uint8)
    cv2.flip(imgResult, 1, imgResult)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) == ord("q"):
        break
