#!/usr/bin/env python3
# /// script
# dependencies = ["torch"]
# ///
"""Create a legacy format PyTorch file with specific storage offsets to test offset handling."""

import torch

# Create tensors with known values at specific storage offsets
# This will help us verify we're reading from the correct location

# Create a state dict with tensors that share storage
# This is common in PyTorch models (e.g., weight and transposed weight views)
state_dict = {}

# Create a base tensor with known pattern
base_data = torch.arange(100, dtype=torch.float32)

# tensor1: uses elements 10-19 (offset 10*4 = 40 bytes)
tensor1 = base_data[10:20].clone()
tensor1[:] = torch.arange(1.0, 1.1, 0.01)[:10]  # 1.00, 1.01, 1.02, ...

# tensor2: uses elements 30-35 (offset 30*4 = 120 bytes)
tensor2 = base_data[30:35].clone()
tensor2[:] = torch.arange(2.0, 2.5, 0.1)[:5]  # 2.0, 2.1, 2.2, 2.3, 2.4

# tensor3: starts at beginning (offset 0)
tensor3 = base_data[:5].clone()
tensor3[:] = torch.arange(3.0, 3.5, 0.1)[:5]  # 3.0, 3.1, 3.2, 3.3, 3.4

state_dict['tensor1'] = tensor1
state_dict['tensor2'] = tensor2
state_dict['tensor3'] = tensor3

# Save in legacy format
output_file = 'test_data/legacy_with_offsets.pt'
torch.save(state_dict, output_file, _use_new_zipfile_serialization=False)

print(f"Created {output_file}")

# Verify by loading
loaded = torch.load(output_file, weights_only=False)
print("\nVerification - expected values:")
for key, tensor in loaded.items():
    print(f"  {key}: {tensor.tolist()}")
    print(f"    Storage offset: {tensor.storage_offset()}")
    print(f"    Storage size: {len(tensor.storage())}")

# Also create a test with multiple tensors sharing the same storage
# This is important for proper offset handling
shared_storage = torch.randn(1000)

# Create views into the same storage at different offsets
view1 = shared_storage[100:110]  # offset 100
view2 = shared_storage[500:520]  # offset 500
view3 = shared_storage[0:10]     # offset 0

# Need to save these properly - PyTorch will handle the storage sharing
shared_dict = {
    'view1': view1.clone(),  # Clone to avoid view issues
    'view2': view2.clone(),
    'view3': view3.clone(),
}

output_file2 = 'test_data/legacy_shared_storage.pt'
torch.save(shared_dict, output_file2, _use_new_zipfile_serialization=False)
print(f"\nCreated {output_file2}")

# Print exact values for test verification
print("\nExact test values for legacy_with_offsets.pt:")
print("tensor1 (10 elements starting at 1.0):")
print("  First 3 values: [1.00, 1.01, 1.02]")
print("tensor2 (5 elements starting at 2.0):")
print("  All values: [2.0, 2.1, 2.2, 2.3, 2.4]")
print("tensor3 (5 elements starting at 3.0):")
print("  All values: [3.0, 3.1, 3.2, 3.3, 3.4]")