import os

import onnx
import onnx_tf
import tensorflow as tf

onnx_model = onnx.load("model.onnx")

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
