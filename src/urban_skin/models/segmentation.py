"""Small U-Net suitable for a first DACL10K baseline."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TinyUNet(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.encoder = DoubleConv(3, 16)
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decoder = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, number_of_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(inputs)
        encoded = self.down(skip)
        decoded = self.up(encoded)
        decoded = torch.cat((decoded, skip), dim=1)
        return self.head(self.decoder(decoded))


class ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
        )
        self.shortcut = (
            nn.Identity()
            if channels_in == channels_out
            else nn.Conv2d(channels_in, channels_out, 1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.convolution(inputs) + self.shortcut(inputs))


class ResidualUNet(nn.Module):
    """A stronger local comparison model with residual blocks and two skip levels."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.first = ResidualBlock(3, 32)
        self.second = ResidualBlock(32, 64)
        self.third = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up_second = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decode_second = ResidualBlock(128, 64)
        self.up_first = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decode_first = ResidualBlock(64, 32)
        self.head = nn.Conv2d(32, number_of_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.first(inputs)
        second = self.second(self.pool(first))
        third = self.third(self.pool(second))
        decoded_second = self.up_second(third)
        decoded_second = self.decode_second(torch.cat((decoded_second, second), dim=1))
        decoded_first = self.up_first(decoded_second)
        decoded_first = self.decode_first(torch.cat((decoded_first, first), dim=1))
        return self.head(decoded_first)


def build_segmentation_model(
    architecture: str, number_of_classes: int
) -> nn.Module:
    if architecture == "tiny_unet":
        return TinyUNet(number_of_classes)
    if architecture == "residual_unet":
        return ResidualUNet(number_of_classes)
    raise ValueError(f"Unknown segmentation architecture: {architecture}")
