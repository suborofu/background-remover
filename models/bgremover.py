import torch
import torch.nn as nn
import torch.nn.functional as F

from models.canny_filter import CannyFilter
from models.refiner import Refiner
from models.unet import UNet


class BGRemover(nn.Module):
    def __init__(
        self,
        body_size=128,
        refiner_size=384,
        body_depth=3,
        refiner_depth=5,
        threshold=5,
        filter_size=5,
        use_refiner=True,
    ):
        super(BGRemover, self).__init__()
        self.body_size = body_size
        self.refiner_size = refiner_size

        self.canny_filter = CannyFilter(threshold=threshold, filter_size=filter_size)
        self.body = UNet(num_classes=2, in_channels=4, depth=body_depth)
        self.refiner = Refiner(kernel_size=11, iterations=refiner_depth)

        self.use_refiner = use_refiner

    def load_state_dict(self, *args, **kwargs):
        self.body.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.body.state_dict(*args, **kwargs)

    def preprocess(self, x):
        edges = self.canny_filter(x)
        x = torch.cat((x, edges), dim=1)
        return x

    def segment_image(self, x):
        shape = x.shape[-2:]
        x = F.interpolate(x, size=(self.body_size, self.body_size), mode="bilinear")
        x = self.body(x)
        x = F.interpolate(x, size=shape, mode="bilinear")
        return x

    def refine_segmentation(self, x, edges):
        if self.use_refiner:
            x1 = self.refiner(x[:, 1:], edges)
            x = torch.cat((1 - x1, x1), dim=1)
        return x

    def postprocess(self, x):
        x = F.softmax(x, dim=1)
        return x

    def forward(self, frame):
        input_shape = frame.shape[-2:]
        frame = F.interpolate(
            frame, size=(self.refiner_size, self.refiner_size), mode="bilinear"
        )
        x1 = self.preprocess(frame)
        x2 = self.segment_image(x1)
        x = self.postprocess(x2)
        x = self.refine_segmentation(x, x1[:, 3:])
        x = F.interpolate(x, size=input_shape, mode="bilinear")
        return x


if __name__ == "__main__":
    model = BGRemover()
    print(model)
    x = torch.randn(3, 3, 360, 640)
    out = model(x)
    print(out.shape)
