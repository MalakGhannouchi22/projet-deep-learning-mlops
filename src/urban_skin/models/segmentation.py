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
