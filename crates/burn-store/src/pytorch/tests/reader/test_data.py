#!/usr/bin/env python3
# /// script
# dependencies = ["torch", "numpy"]
# ///
"""
Generate test PyTorch .pt files for testing the burn-store PyTorch reader.
Run with: uv run test_files.py
"""

import torch
import numpy as np
import os
from pathlib import Path

# Create test directory
test_dir = Path(__file__).parent / "test_data"
test_dir.mkdir(exist_ok=True)

def save_test_file(filename, data, description):
    """Save a test file and print what was saved."""
    filepath = test_dir / filename
    torch.save(data, filepath)
    print(f"✓ {filename}: {description}")
    return filepath

# Test 1: Simple tensors of different types
print("\n=== Generating Basic Tensor Tests ===")

# Float32 tensor (wrap in dict for compatibility)
float32_tensor = torch.tensor([1.0, 2.5, -3.7, 0.0], dtype=torch.float32)
save_test_file("float32.pt", {"tensor": float32_tensor}, "Float32 tensor [1.0, 2.5, -3.7, 0.0]")

# Float64 tensor
float64_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float64)
save_test_file("float64.pt", {"tensor": float64_tensor}, "Float64 tensor [1.1, 2.2, 3.3]")

# Int64 tensor
int64_tensor = torch.tensor([100, -200, 300, 0], dtype=torch.int64)
save_test_file("int64.pt", {"tensor": int64_tensor}, "Int64 tensor [100, -200, 300, 0]")

# Int32 tensor
int32_tensor = torch.tensor([10, 20, -30], dtype=torch.int32)
save_test_file("int32.pt", {"tensor": int32_tensor}, "Int32 tensor [10, 20, -30]")

# Int16 tensor
int16_tensor = torch.tensor([1000, -2000, 3000], dtype=torch.int16)
save_test_file("int16.pt", {"tensor": int16_tensor}, "Int16 tensor [1000, -2000, 3000]")

# Int8 tensor
int8_tensor = torch.tensor([127, -128, 0, 50], dtype=torch.int8)
save_test_file("int8.pt", {"tensor": int8_tensor}, "Int8 tensor [127, -128, 0, 50]")

# Boolean tensor
bool_tensor = torch.tensor([True, False, True, True, False], dtype=torch.bool)
save_test_file("bool.pt", {"tensor": bool_tensor}, "Bool tensor [True, False, True, True, False]")

# Float16 tensor (half precision)
float16_tensor = torch.tensor([1.5, -2.25, 3.125], dtype=torch.float16)
save_test_file("float16.pt", {"tensor": float16_tensor}, "Float16 tensor [1.5, -2.25, 3.125]")

# BFloat16 tensor
bfloat16_tensor = torch.tensor([1.5, -2.5, 3.5], dtype=torch.bfloat16)
save_test_file("bfloat16.pt", {"tensor": bfloat16_tensor}, "BFloat16 tensor [1.5, -2.5, 3.5]")

# Test 2: Multi-dimensional tensors
print("\n=== Generating Multi-dimensional Tensor Tests ===")

# 2D tensor
tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
save_test_file("tensor_2d.pt", {"tensor": tensor_2d}, "2D tensor shape (3, 2)")

# 3D tensor
torch.manual_seed(42)
tensor_3d = torch.randn(2, 3, 4) * 10
save_test_file("tensor_3d.pt", {"tensor": tensor_3d}, "3D tensor shape (2, 3, 4)")

# 4D tensor (common for conv weights)
tensor_4d = torch.randn(2, 3, 2, 2)
save_test_file("tensor_4d.pt", {"tensor": tensor_4d}, "4D tensor shape (2, 3, 2, 2)")

# Test 3: State dict (multiple tensors)
print("\n=== Generating State Dict Tests ===")

state_dict = {
    "weight": torch.randn(3, 4),
    "bias": torch.randn(3),
    "running_mean": torch.zeros(3),
    "running_var": torch.ones(3),
}
save_test_file("state_dict.pt", state_dict, "State dict with 4 tensors")

# Nested state dict
nested_dict = {
    "layer1": {
        "weight": torch.randn(2, 3),
        "bias": torch.randn(2)
    },
    "layer2": {
        "weight": torch.randn(4, 2),
        "bias": torch.randn(4)
    }
}
save_test_file("nested_dict.pt", nested_dict, "Nested state dict")

# Test 4: Model checkpoint format
print("\n=== Generating Model Checkpoint Tests ===")

# Typical checkpoint format (use string keys for compatibility)
checkpoint = {
    "model_state_dict": {
        "fc1.weight": torch.randn(10, 5),
        "fc1.bias": torch.randn(10),
        "fc2.weight": torch.randn(3, 10),
        "fc2.bias": torch.randn(3),
    },
    "optimizer_state_dict": {
        "state": {
            "0": {  # Use string key instead of integer
                "momentum_buffer": torch.randn(10, 5)
            }
        }
    },
    "epoch": 42,
    "loss": 0.123
}
save_test_file("checkpoint.pt", checkpoint, "Full checkpoint with model and optimizer state")

# Test 5: Edge cases
print("\n=== Generating Edge Case Tests ===")

# Empty tensor (1D with 0 elements)
empty_tensor = torch.zeros(0)
save_test_file("empty.pt", {"tensor": empty_tensor}, "Empty tensor")

# Scalar tensor (0-dimensional)
scalar_tensor = torch.tensor(42.0)
save_test_file("scalar.pt", {"tensor": scalar_tensor}, "Scalar tensor (0-dim)")

# Large shape but small data (testing shape vs actual data)
sparse_like = torch.zeros(100, 100)
sparse_like[0, 0] = 1.0
sparse_like[50, 50] = 2.0
sparse_like[99, 99] = 3.0
save_test_file("large_shape.pt", {"tensor": sparse_like}, "Large shape (100, 100) mostly zeros")

# Test 6: Mixed types in dict
print("\n=== Generating Mixed Type Tests ===")

mixed_types = {
    "float32": torch.tensor([1.0, 2.0], dtype=torch.float32),
    "int64": torch.tensor([100, 200], dtype=torch.int64),
    "bool": torch.tensor([True, False], dtype=torch.bool),
    "float64": torch.tensor([1.1, 2.2], dtype=torch.float64),
}
save_test_file("mixed_types.pt", mixed_types, "Dict with mixed tensor types")

# Test 7: Special values
print("\n=== Generating Special Value Tests ===")

# NaN and Inf values
special_values = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0, 1.0])
save_test_file("special_values.pt", {"tensor": special_values}, "Tensor with NaN and Inf")

# Very small and very large values
extreme_values = torch.tensor([1e-30, 1e30, -1e-30, -1e30], dtype=torch.float32)
save_test_file("extreme_values.pt", {"tensor": extreme_values}, "Tensor with extreme values")

# Test 8: Parameter wrapper (common in models)
print("\n=== Generating Parameter Tests ===")

import torch.nn as nn
param = nn.Parameter(torch.randn(3, 3))
param_dict = {"param": param}
save_test_file("parameter.pt", param_dict, "nn.Parameter wrapped tensor")

# Test 9: Buffer-style tensors
print("\n=== Generating Buffer Tests ===")

# Simulate model buffers
buffers = {
    "buffer1": torch.tensor([1, 2, 3], dtype=torch.int32),
    "buffer2": torch.tensor([True, False], dtype=torch.bool),
}
save_test_file("buffers.pt", buffers, "Model buffers")

# Test 10: Complex nested structure
print("\n=== Generating Complex Structure Tests ===")

complex_structure = {
    "metadata": {
        "version": 1,
        "name": "test_model"
    },
    "state": {
        "encoder": {
            "layer_0": {
                "weight": torch.randn(4, 3),
                "bias": torch.randn(4)
            },
            "layer_1": {
                "weight": torch.randn(2, 4),
                "bias": torch.randn(2)
            }
        },
        "decoder": {
            "weight": torch.randn(3, 2),
            "bias": torch.randn(3)
        }
    },
    "config": {
        "hidden_size": 4,
        "num_layers": 2
    }
}
save_test_file("complex_structure.pt", complex_structure, "Complex nested structure")

print(f"\n✅ Generated {len(list(test_dir.glob('*.pt')))} test files in {test_dir}")
print("\nTest files can be used to verify PyTorch reader functionality:")
print("- Different data types (float32, int64, bool, etc.)")
print("- Multi-dimensional tensors")
print("- State dicts and nested structures")
print("- Edge cases (empty, scalar, special values)")
print("- Model checkpoints and parameters")
