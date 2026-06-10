#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 8 * 8, 16)  # Changed for smaller input size

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    # Use a smaller input size
    # 1 batch, 3 channels (RGB), 8x8 image (small input)
    x1 = torch.ones(1, 3, 8, 8)
    _ = model(x1)
    model.eval()  # Set to eval mode to freeze running stats
    # Save the model to safetensors after the first forward
    save_file(model.state_dict(), "multi_layer.safetensors")

    x2 = torch.ones(1, 3, 8, 8)
    print("Input shape: {}", x2.shape)
    output = model(x2)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)


if __name__ == "__main__":
    main()
