#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, (2,2))

    def forward(self, x):
        x = self.conv1(x)
        return x


def main():
    torch.set_printoptions(precision=8)
    torch.manual_seed(1)
    model = Model().to(torch.device("cpu"))
    torch.save({"my_state_dict": model.state_dict()}, "top_level_key.pt")

if __name__ == '__main__':
    main()
