#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.norm1 = nn.GroupNorm(2, 6)
        
    def forward(self, x):
        x = self.norm1(x)
        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    torch.save(model.state_dict(), "group_norm.pt")
    
    x2 = torch.rand(1, 6, 2, 2)
    print("Input shape: {}", x2.shape)
    print("Input: {}", x2)
    output = model(x2)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
