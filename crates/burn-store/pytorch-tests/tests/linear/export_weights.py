#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 4, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # Add relu so that PyTorch optimizer does not combine fc1 and fc2
        x = self.fc2(x)

        return x


class ModelWithBias(nn.Module):
    def __init__(self):
        super(ModelWithBias, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        
    def forward(self, x):
        x = self.fc1(x)

        return x


def main():

    torch.set_printoptions(precision=8)
    torch.manual_seed(1)

    model = Model().to(torch.device("cpu"))
    model_with_bias = ModelWithBias().to(torch.device("cpu"))

    torch.save(model.state_dict(), "linear.pt")
    torch.save(model_with_bias.state_dict(), "linear_with_bias.pt")
    
    input = torch.rand(1, 2, 2, 2)
    print("Input shape: {}", input.shape)
    print("Input: {}", input)

    output = model(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    print("Model with bias")
    output = model_with_bias(input)
    print("Output: {}", output)
    print("Output Shape: {}", output.shape)

    


if __name__ == '__main__':
    main()
