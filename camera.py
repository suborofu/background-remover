# importing the modules
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

providers = [
    # "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

ort_session = ort.InferenceSession("model.onnx", providers=providers)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# set Width and Height of output Screen
frameWidth = 640
frameHeight = 360

# capturing Video from Webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# set brightness, id is 10 and
# value can be changed accordingly
cap.set(10, 150)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    img2 = np.array(imgResult)[np.newaxis, ...].astype(np.float32) / 255
    img2 = np.transpose(img2, [0, 3, 1, 2])
    img2 = np.stack((img2[:, 0], img2[:, 1], img2[:, 2], img2[:, 0]), axis=1)

    out = ort_session.run([output_name], {input_name: img2})[0]
    out = F.softmax(torch.from_numpy(out), dim=1).numpy()[:, 1]
    out = (out[0].round() * 255).astype(np.uint8)
    out = np.stack((out, out, out), axis=-1)

    blured = np.zeros_like(out)
    imgResult[out == 0] = blured[out == 0]
    cv2.flip(imgResult, 1, imgResult)
    # displaying output on Screen
    cv2.imshow("Result", imgResult)

    # condition to break programs execution
    # press q to stop the execution of program
    if cv2.waitKey(1) == ord("q"):
        break
