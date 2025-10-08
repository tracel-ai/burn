#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "safetensors",
#     "packaging",
#     "numpy",
# ]
# ///
"""
Generate a large model (~312MB) in both PyTorch and SafeTensors formats for unified benchmarking.

Usage:
    uv run benches/generate_unified_models.py

The script will create model files in /tmp/simple_bench_models/ directory.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
import tempfile
from safetensors.torch import save_file

def get_temp_dir():
    """Get the appropriate temp directory."""
    temp_dir = Path(tempfile.gettempdir()) / "simple_bench_models"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

class LargeModel(nn.Module):
    """Large model with 20 layers to match Rust benchmark."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

        # Create a model with 20 layers matching the Rust LargeModel
        for i in range(20):
            in_size = 1024 if i == 0 else 2048
            out_size = 2048
            self.layers.append(nn.Linear(in_size, out_size))

        print(f"Created model with {len(self.layers)} layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def calculate_model_size(model):
    """Calculate the size of the model in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    return total_params, size_mb

def initialize_weights(model):
    """Initialize model weights with random values."""
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

def save_pytorch_format(model, output_dir):
    """Save model in PyTorch format."""
    pt_path = output_dir / "large_model.pt"

    # Save as checkpoint with model_state_dict (common format)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'model_type': 'large_benchmark_model',
            'num_layers': len(model.layers),
        }
    }
    torch.save(checkpoint, pt_path)

    return pt_path

def save_safetensors_format(model, output_dir):
    """Save model in SafeTensors format."""
    st_path = output_dir / "large_model.safetensors"

    # Convert state dict to safetensors format
    state_dict = model.state_dict()
    # Ensure all tensors are contiguous and on CPU
    state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}

    # Save with metadata
    metadata = {
        'model_type': 'large_benchmark_model',
        'num_layers': str(len(model.layers)),
    }
    save_file(state_dict, st_path, metadata=metadata)

    return st_path

def verify_files(pt_path, st_path):
    """Verify the saved files can be loaded."""
    # Verify PyTorch file
    checkpoint = torch.load(pt_path, map_location='cpu')
    pt_keys = set(checkpoint['model_state_dict'].keys())
    print(f"  PyTorch file: {len(pt_keys)} tensors")

    # Verify SafeTensors file
    from safetensors import safe_open
    with safe_open(st_path, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())
        print(f"  SafeTensors file: {len(st_keys)} tensors")

    # Check keys match
    if pt_keys != st_keys:
        print("  ⚠️ Warning: Keys don't match between formats!")
    else:
        print("  ✓ Keys match between formats")

def main():
    print("🔧 Generating unified benchmark model files...")
    print("")

    output_dir = get_temp_dir()
    print(f"📁 Output directory: {output_dir}")
    print("")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create the large model
    print("📝 Creating large model...")
    model = LargeModel()

    # Calculate and display model size
    total_params, size_mb = calculate_model_size(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {size_mb:.2f} MB")
    print("")

    # Initialize weights
    print("🎲 Initializing weights...")
    initialize_weights(model)

    # Save in PyTorch format
    print("💾 Saving PyTorch format...")
    pt_path = save_pytorch_format(model, output_dir)
    pt_size_mb = pt_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {pt_path}")
    print(f"  File size: {pt_size_mb:.2f} MB")
    print("")

    # Save in SafeTensors format
    print("💾 Saving SafeTensors format...")
    st_path = save_safetensors_format(model, output_dir)
    st_size_mb = st_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {st_path}")
    print(f"  File size: {st_size_mb:.2f} MB")
    print("")

    # Verify files
    print("🔍 Verifying saved files...")
    verify_files(pt_path, st_path)
    print("")

    print(f"✅ Model files generated successfully!")
    print("")
    print("📊 Summary:")
    print(f"  PyTorch file: {pt_path.name} ({pt_size_mb:.2f} MB)")
    print(f"  SafeTensors file: {st_path.name} ({st_size_mb:.2f} MB)")
    print("")
    print("💡 To run the unified benchmark:")
    print("   cargo bench --bench unified_loading")

if __name__ == "__main__":
    main()