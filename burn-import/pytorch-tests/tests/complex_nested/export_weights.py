#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(2, 4, (3, 2)),
            ConvBlock(4, 6, (3, 2)),
        )
        self.norm1 = nn.BatchNorm2d(6)

        self.fc1 = nn.Linear(120, 12)
        self.fc2 = nn.Linear(12, 10)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.norm1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(2)


    model = Net().to(torch.device("cpu"))

    # Condition the model (batch norm requires a forward pass to compute the mean and variance)
    x1 = torch.ones(1, 2, 9, 6) - 0.1
    x2 = torch.ones(1, 2, 9, 6) - 0.3
    output = model(x1)
    output = model(x2)
    model.eval() # set to eval mode

    torch.save(model.state_dict(), "complex_nested.pt")

    # feed test data
    x = torch.ones(1, 2, 9, 6) - 0.5
    output = model(x)
    print("Input shape: {}", x.shape)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
