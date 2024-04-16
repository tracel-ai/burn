#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.norm1 = nn.BatchNorm2d(5)
        
    def forward(self, x):
        x = self.norm1(x)
        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    # Condition batch norm (each forward will affect the running stats)
    x1 = torch.ones(1, 5, 2, 2) - 0.5
    _ = model(x1)
    model.eval() # Set to eval mode to freeze running stats
    # Save the model after the first forward
    torch.save(model.state_dict(), "batch_norm2d.pt")
    
    x2 = torch.ones(1, 5, 2, 2) - 0.3
    print("Input shape: {}", x2.shape)
    output = model(x2)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
