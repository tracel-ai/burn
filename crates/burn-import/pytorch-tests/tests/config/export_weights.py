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

CONFIG = {
    "n_head": 2,
    "n_layer": 3,
    "d_model": 512,
    "some_float": 0.1,
    "some_int": 1,
    "some_bool": True,
    "some_str": "hello",
    "some_list_int": [1, 2, 3],
    "some_list_str": ["hello", "world"],
    "some_list_float": [0.1, 0.2, 0.3],
    "some_dict": {
        "some_key": "some_value"
    }
}

class ModelWithBias(nn.Module):
    def __init__(self):
        super(ModelWithBias, self).__init__()
        self.fc1 = nn.Linear(2, 3)

    def forward(self, x):
        x = self.fc1(x)

        return x


def main():

    model = Model().to(torch.device("cpu"))

    weights_with_config = {
        "my_model": model.state_dict(),
        "my_config": CONFIG
    }

    torch.save(weights_with_config, "weights_with_config.pt")


if __name__ == '__main__':
    main()
