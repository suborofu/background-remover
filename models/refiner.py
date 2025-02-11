import torch.nn as nn
import torch.nn.functional as F

from models.modules import Dilation2D, DirectedDilation2D, DirectedErosion2D


def pad(x, pad_size):
    return F.pad(x, [pad_size, pad_size, pad_size, pad_size], "constant", 0)


def unpad(x, pad_size):
    return x[:, :, pad_size:-pad_size, pad_size:-pad_size]


class Refiner(nn.Module):
    def __init__(self, kernel_size=5, iterations=5, threshold=0.7):
        super(Refiner, self).__init__()
        self.dilation1 = Dilation2D(kernel_size, iterations)
        self.dilation2 = DirectedDilation2D(kernel_size, iterations, direction="down")
        self.erosion2 = DirectedErosion2D(kernel_size, iterations, direction="up")
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.threshold = threshold

    def forward(self, x, edges):
        x1 = (x > self.threshold).float()
        x1 = self.dilation1(x1)
        x2 = (x1 * edges + x).clip(min=0, max=1)
        x2 = (x2 > self.threshold).float()

        pad_size = self.kernel_size // 2 * self.iterations
        x2 = pad(x2, pad_size)
        x2 = self.dilation2(x2)
        x2 = self.erosion2(x2)
        x2 = unpad(x2, pad_size)
        return x2
