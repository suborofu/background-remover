import torch

from models.bgremover import BGRemover

model = BGRemover(body_depth=4, refiner_depth=2)
# load_file = (
#     config.CHECKPOINT_PATH + "/" + sorted(os.listdir(config.CHECKPOINT_PATH))[-1]
# )
# state_dict = torch.load(load_file)
# model.load_state_dict(state_dict)
model = model.eval()

dummy_input = torch.randn(1, 3, 360, 640)
print(model(dummy_input))
# exported = torch.export.export(
#     model,
#     (dummy_input,),
#     dynamic_shapes=[
#         {
#             2: torch.export.Dim("height", min=128),
#             3: torch.export.Dim("width", min=128),
#         },
#     ],
# )
# script_module = torch.jit.script(model)
onnx_program = torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
)
# onnx_program.save("model.onnx")
