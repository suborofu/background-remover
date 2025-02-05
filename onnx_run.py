import time

import numpy as np
import onnxruntime as ort

providers = [
    # "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

ort_session = ort.InferenceSession("model.onnx", providers=providers)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

input = np.random.randn(1, 3, 720, 1280).astype(np.float32)

t = time.time()

for i in range(60):
    result = ort_session.run([output_name], {input_name: input})
print(60 / (time.time() - t))

print(result[0].shape)
