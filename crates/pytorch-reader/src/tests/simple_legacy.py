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

print("\nCreated simple_legacy.pt")

# Verify
loaded = torch.load('test_data/simple_legacy.pt', weights_only=False)
print(f"Loaded {len(loaded)} tensors")
for key, val in loaded.items():
    print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
    
# The storage has 100 elements while each tensor only uses 10
base = torch.arange(100, dtype=torch.float32)  
tensor1 = base[10:20]  
tensor2 = base[50:60]  
torch.save({'tensor1': tensor1, 'tensor2': tensor2}, 'test_data/legacy_uncloned_views.pt',  
           _use_new_zipfile_serialization=False)

# Verify
print("\nCreated legacy_uncloned_views.pt")
loaded2 = torch.load('test_data/legacy_uncloned_views.pt', weights_only=False)
print(f"Loaded {len(loaded2)} tensors")
for key, val in loaded2.items():
    print(f"  {key}: shape {val.shape}, dtype {val.dtype}")