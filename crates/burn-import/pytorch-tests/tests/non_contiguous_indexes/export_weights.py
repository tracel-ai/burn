#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        num_layers = 5  # Number of repeated convolutional layers

        # Create a list to store the layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))

        # Use nn.Sequential to create a single module from the layers
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x

def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    torch.save(model.state_dict(), "non_contiguous_indexes.pt")

    input = torch.rand(1, 2, 5, 5)
    print("Input shape: {}", input.shape)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

if __name__ == '__main__':
    main()
