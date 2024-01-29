#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(10, 3)
        
    def forward(self, x):
        x = self.embed(x)
        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))

    torch.save(model.state_dict(), "embedding.pt")
    
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print("Input shape: {}", input.shape)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
