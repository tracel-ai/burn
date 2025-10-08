#!/usr/bin/env python3
# /// script
# dependencies = ["torch"]
# ///
"""Create a simple legacy format PyTorch file."""

import torch

# Create a simple state dict
state_dict = {
    'weight': torch.randn(2, 3),
    'bias': torch.ones(2),
    'running_mean': torch.zeros(2),
}

# Save without using zip format (legacy format)
torch.save(state_dict, 'test_data/simple_legacy.pt', _use_new_zipfile_serialization=False)

print("Created simple_legacy.pt")

# Verify
loaded = torch.load('test_data/simple_legacy.pt', weights_only=False)
print(f"Loaded {len(loaded)} tensors")
for key, val in loaded.items():
    print(f"  {key}: shape {val.shape}, dtype {val.dtype}")