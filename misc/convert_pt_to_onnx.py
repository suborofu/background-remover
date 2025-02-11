import os

import onnx
import onnx_tf
import tensorflow as tf
import torch
from onnxsim import simplify

import config
from models.bgremover import BGRemover

model = BGRemover(
    config.BODY_SIZE,
    config.REFINER_SIZE,
    config.BODY_DEPTH,
    config.REFINER_DEPTH,
    config.THRESHOLD,
    config.FILTER_SIZE,
    use_refiner=False,
)
load_file = (
    config.CHECKPOINT_PATH + "/" + sorted(os.listdir(config.CHECKPOINT_PATH))[-1]
)
state_dict = torch.load(load_file)
model.load_state_dict(state_dict)

model = model.eval()

dummy_input = torch.rand(1, 3, config.REFINER_SIZE, config.REFINER_SIZE)
print(model(dummy_input))

for p in model.parameters():
    p.requires_grad = False

with torch.no_grad():
    onnx_program = torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={
        #     "input": {0: "batch", 2: "height", 3: "width"},
        #     "output": {0: "batch", 2: "height", 3: "width"},
        # },
        opset_version=12,
    )

onnx_model = onnx.load("model.onnx")
onnx_model, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(onnx_model, "model.onnx")

tf_model_path = "model.pb"
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

command = "tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve model.pb model.tfjs"
os.system(command)
