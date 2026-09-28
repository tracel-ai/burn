#!/usr/bin/env python3

import torch
import torch.nn as nn


class Flip(nn.Module):
    """Parameter-free layer, like the Flip between coupling layers in VITS."""

    def forward(self, x):
        return torch.flip(x, [1])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # flows.0, flows.2, flows.4 have parameters; flows.1, flows.3 do not.
        # The Burn side mirrors these indices directly, so they must NOT be renumbered.
        flows = []
        for _ in range(3):
            flows.append(nn.Conv1d(2, 2, kernel_size=1, bias=True))
            flows.append(Flip())
        self.flows = nn.ModuleList(flows)

        # fc.0, fc.2 have parameters; fc.1 is ReLU.
        # The Burn side is a Vec<Linear> of length 2, so these DO need renumbering.
        self.fc = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))

    def forward(self, x):
        for flow in self.flows:
            x = flow(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x.transpose(1, 2)


def main():
    torch.set_printoptions(precision=8)
    torch.manual_seed(2)

    model = Model().to(torch.device("cpu"))

    for name, param in model.state_dict().items():
        print(name, list(param.shape))

    torch.save(model.state_dict(), "keep_indices.pt")

    input = torch.rand(1, 2, 3)
    print("Input: {}", input)
    output = model(input)
    print("Output: {}", output)


if __name__ == "__main__":
    main()
