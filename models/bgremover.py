import torch
import torch.nn as nn
import torch.nn.functional as F

from models.canny_filter import CannyFilter
from models.modules import Skip
from models.unet import UNet


class BGRemover(nn.Module):
    def __init__(
        self,
        body_size=128,
        refiner_size=384,
        body_depth=3,
        refiner_depth=2,
        threshold=5,
        filter_size=5,
    ):
        super(BGRemover, self).__init__()
        self.body_size = body_size
        self.refiner_size = refiner_size

        self.canny_filter = CannyFilter(threshold=threshold, filter_size=filter_size)
        self.body = UNet(num_classes=2, in_channels=4, depth=body_depth)
        self.refiner = Skip()

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

    def refine_segmentation(self, x):
        x = self.refiner(x)
        return x

    def postprocess(self, x):
        x = F.softmax(x, dim=1)[:, 1]
        return x

    def forward(self, frame):
        input_shape = frame.shape[-2:]
        frame = F.interpolate(frame, size=(self.refiner_size, self.refiner_size))
        x = self.preprocess(frame)
        x = self.segment_image(x)
        x = self.refine_segmentation(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear")
        x = self.postprocess(x)
        return x


if __name__ == "__main__":
    model = BGRemover()
    x = torch.randn(3, 3, 360, 640)
    out = model(x)
    print(out)
    print(out.shape)
