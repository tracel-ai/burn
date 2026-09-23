#!/usr/bin/env python3
# /// script
# dependencies = ["torch"]
# ///
"""
Create full-model saves, `torch.save(model)` rather than `torch.save(model.state_dict())`.

The model exercises the rules the reader applies when it rebuilds `state_dict()` from the
pickled module: a parameter slot holding None (`bias=False`), children under numeric names
(`nn.Sequential`), a child registered under a second name (pickled once, then referenced
from the memo), a persistent and a non-persistent buffer, and a tensor assigned as a plain
attribute, which is not a buffer and so is not part of the state dict. The int64 buffer
(`num_batches_tracked`) checks that the element type survives.

  full_model.pt             torch.save(model) at the default protocol (2), where the
                            non-persistent buffer set is a REDUCE of set under its
                            Python 2 module name, __builtin__
  full_model_state_dict.pt  torch.save(model.state_dict()) of the same model, which the
                            full-model save must load identically to
  full_model_checkpoint.pt  the model under a key beside other values, at protocol 4,
                            where the set is written with EMPTY_SET and ADDITEMS

Run with: uv run create_full_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn

test_dir = Path(__file__).parent / "test_data"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2, bias=False)
        self.blocks = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.tied = self.fc
        self.register_buffer("running_mean", torch.arange(3, dtype=torch.float32))
        self.register_buffer("mask", torch.ones(2, dtype=torch.bool), persistent=False)
        self.scale = torch.tensor([2.0])

    def forward(self, x):
        return self.blocks(self.head(self.fc(x)))


def main():
    torch.manual_seed(0)
    model = Net()

    torch.save(model, test_dir / "full_model.pt")
    torch.save(model.state_dict(), test_dir / "full_model_state_dict.pt")
    torch.save(
        {"model": model, "epoch": 3},
        test_dir / "full_model_checkpoint.pt",
        pickle_protocol=4,
    )
    for name, tensor in model.state_dict().items():
        print(f"{name}: {tuple(tensor.shape)} {tensor.dtype}")


if __name__ == "__main__":
    main()
