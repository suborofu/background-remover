import time

import numpy as np
import onnxruntime as ort

import config

providers = [
    # "CUDAExecutionProvider",
    # "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

ort_session = ort.InferenceSession("model.onnx", providers=providers)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print(input_name, output_name)

input = np.random.randn(1, 3, config.REFINER_SIZE, config.REFINER_SIZE).astype(
    np.float32
)
input[:] = 0
ort_session.run([output_name], {input_name: input})
t = time.time()
for i in range(100):
    result = ort_session.run([output_name], {input_name: input})
print(100 / (time.time() - t))

print(result[0].shape)
