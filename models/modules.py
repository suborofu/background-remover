import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        padding=None,
        stride=1,
        dilation=1,
    ):
        super(Conv2D, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
            groups=in_channels,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = Conv2D(self.in_channels, self.out_channels, kernel_size=3)
        self.conv2 = Conv2D(self.out_channels, self.out_channels, kernel_size=3)

        # if self.pooling:
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            Conv2D(in_channels, out_channels, kernel_size=1),
        )

        self.conv1 = Conv2D(2 * self.out_channels, self.out_channels, kernel_size=3)
        self.conv2 = Conv2D(self.out_channels, self.out_channels, kernel_size=3)

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AdaptiveAvgPool2D(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = torch.tensor(output_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        """
        shape_x = x.shape
        if shape_x[-1] < self.output_size[-1]:
            paddzero = torch.zeros(
                (shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1])
            )
            x = torch.cat((x, paddzero), axis=-1)

        stride_size = torch.floor(torch.tensor(x.shape[-2:]) / self.output_size).to(
            dtype=torch.int32
        )
        kernel_size = torch.tensor(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(
            kernel_size=kernel_size.tolist(), stride=kernel_size.tolist()
        )
        x = avg(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.avgpool = nn.Sequential(
            AdaptiveAvgPool2D((2, 2)),
            Conv2D(in_channels, out_channels, kernel_size=1, padding=0),
        )

        self.conv1 = Conv2D(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1
        )
        self.conv2 = Conv2D(
            in_channels, out_channels, kernel_size=3, padding=6, dilation=6
        )
        self.conv3 = Conv2D(
            in_channels, out_channels, kernel_size=3, padding=12, dilation=12
        )
        self.conv4 = Conv2D(
            in_channels, out_channels, kernel_size=3, padding=18, dilation=18
        )

        self.conv5 = Conv2D(
            out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1
        )

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.conv5(xc)

        return y


class Skip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
