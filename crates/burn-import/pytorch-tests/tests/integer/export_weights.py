#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        buffer = torch.tensor([1, 2, 3])
        self.register_buffer("buffer", buffer, persistent=True)
        
    def forward(self, x):
        x = self.buffer
        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    torch.save(model.state_dict(), "integer.pt")
    
    input = torch.ones(3, 3)
    print("Input shape: {}", input.shape)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
