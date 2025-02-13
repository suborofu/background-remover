import torch
import torch.nn as nn


def gaussian(size, std=1.0):
    start = -(size - 1) / 2.0

    constant = 1 / (std * 2**0.5)

    k = torch.linspace(
        start=start * constant, end=(start + (size - 1)) * constant, steps=size
    )

    return torch.exp(-(k**2))


class CannyFilter(nn.Module):
    def __init__(self, threshold=10, filter_size=5):
        super(CannyFilter, self).__init__()
        self.threshold = torch.tensor(threshold)
        generated_filters = gaussian(filter_size, std=1.0).unsqueeze(0)

        self.gaussian_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, filter_size),
            padding=(0, filter_size // 2),
        )
        self.gaussian_filter_horizontal.weight.data.copy_(generated_filters)
        self.gaussian_filter_horizontal.bias.data.copy_(torch.tensor([0.0]))
        self.gaussian_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(filter_size, 1),
            padding=(filter_size // 2, 0),
        )
        self.gaussian_filter_vertical.weight.data.copy_(generated_filters.T)
        self.gaussian_filter_vertical.bias.data.copy_(torch.tensor([0.0]))

        sobel_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)
        self.sobel_filter_horizontal.bias.data.copy_(torch.tensor([0.0]))
        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=sobel_filter.shape,
            padding=sobel_filter.shape[0] // 2,
        )
        self.sobel_filter_vertical.weight.data.copy_(sobel_filter.T)
        self.sobel_filter_vertical.bias.data.copy_(torch.tensor([0.0]))

        # filters were flipped manually
        filter_0 = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        filter_45 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        filter_90 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        filter_135 = torch.tensor([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
        filter_180 = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        filter_225 = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        filter_270 = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        filter_315 = torch.tensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        all_filters = torch.stack(
            [
                filter_0,
                filter_45,
                filter_90,
                filter_135,
                filter_180,
                filter_225,
                filter_270,
                filter_315,
            ]
        )

        self.directional_filter = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=filter_0.shape,
            padding=filter_0.shape[-1] // 2,
        )
        self.directional_filter.weight.data.copy_(all_filters.unsqueeze(1))
        self.directional_filter.bias.data.copy_(torch.zeros(all_filters.shape[0]))

        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super(CannyFilter, self).train(False)
        return self

    def forward(self, img):
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2 + 1e-8)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2 + 1e-8)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2 + 1e-8)
        grad_orientation = torch.atan2(
            grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b
        ) * (180.0 / 3.141592653589793)
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.shape[2]
        width = inidices_positive.shape[3]
        pixel_count = height * width
        pixel_range = torch.arange(pixel_count).to(img.device)

        indices = inidices_positive.flatten(1) * pixel_count + pixel_range
        indices = indices.round().long()
        indices[indices < 0] = 0
        channel_select_filtered_positive = (
            all_filtered.flatten(1).gather(1, indices).view(-1, 1, height, width)
        )

        indices = inidices_negative.flatten(1) * pixel_count + pixel_range
        indices = indices.round().long()
        indices[indices < 0] = 0
        channel_select_filtered_negative = (
            all_filtered.flatten(1).gather(1, indices).view(-1, 1, height, width)
        )

        channel_select_filtered = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative], dim=1
        )

        is_max = channel_select_filtered.min(dim=1)[0] > 0.0

        grad_mag[is_max == 0] = 0.0
        grad_mag[grad_mag < self.threshold] = 0.0

        grad_mag[:, :, 0, :] = 0.0
        grad_mag[:, :, -1, :] = 0.0
        grad_mag[:, :, :, 0] = 0.0
        grad_mag[:, :, :, -1] = 0.0
        return grad_mag
