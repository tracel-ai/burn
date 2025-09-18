#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
# ]
# ///
"""
Generate PyTorch model files for benchmarking.

This script creates PyTorch model files (.pth) of various sizes for benchmarking
the PyTorch loading functionality in burn-store.

Usage:
    uv run benches/generate_pytorch_models.py

The script will create model files in /tmp/pytorch_bench_models/ directory.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
import tempfile
import platform

def get_temp_dir():
    """Get the appropriate temp directory based on OS."""
    if platform.system() == "Darwin":  # macOS
        # Use /var/folders on macOS as it's more reliable
        base_tmp = tempfile.gettempdir()
    else:
        base_tmp = "/tmp"

    temp_dir = Path(base_tmp) / "pytorch_bench_models"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

class SimpleModel(nn.Module):
    """Simple model matching the Rust SimpleModel structure."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1024)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class MediumModel(nn.Module):
    """Medium model with various layer types matching the Rust MediumModel."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

    def forward(self, x):
        # Forward pass implementation (not used for benchmarking)
        pass

class LargeModel(nn.Module):
    """Large model with 20 layers matching the Rust LargeModel."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(20):
            in_size = 1024 if i == 0 else 2048
            self.layers.append(nn.Linear(in_size, 2048))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def initialize_weights(model):
    """Initialize model weights with random values."""
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

def save_model_formats(model, base_path):
    """Save model in different PyTorch formats."""
    # Save as state_dict (most common format)
    torch.save(model.state_dict(), f"{base_path}_state_dict.pth")

    # Save as full model checkpoint with metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 100,
        'optimizer_state_dict': None,
        'loss': 0.001,
    }
    torch.save(checkpoint, f"{base_path}_checkpoint.pth")

    # Save with top-level key (common in HuggingFace models)
    wrapped = {
        'state_dict': model.state_dict(),
        'config': {'model_type': 'benchmark_model'},
    }
    torch.save(wrapped, f"{base_path}_wrapped.pth")

def main():
    print("🔧 Generating PyTorch model files for benchmarking...")

    temp_dir = get_temp_dir()
    print(f"📁 Output directory: {temp_dir}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate Simple Model files
    print("\n📝 Creating Simple Model files...")
    simple_model = SimpleModel()
    initialize_weights(simple_model)
    save_model_formats(simple_model, temp_dir / "simple_model")

    # Calculate approximate size
    simple_size = sum(p.numel() * 4 for p in simple_model.parameters()) / (1024 * 1024)
    print(f"   ✓ Simple model size: ~{simple_size:.2f} MB")

    # Generate Medium Model files
    print("\n📝 Creating Medium Model files...")
    medium_model = MediumModel()
    initialize_weights(medium_model)
    save_model_formats(medium_model, temp_dir / "medium_model")

    medium_size = sum(p.numel() * 4 for p in medium_model.parameters()) / (1024 * 1024)
    print(f"   ✓ Medium model size: ~{medium_size:.2f} MB")

    # Generate Large Model files
    print("\n📝 Creating Large Model files...")
    large_model = LargeModel()
    initialize_weights(large_model)
    save_model_formats(large_model, temp_dir / "large_model")

    large_size = sum(p.numel() * 4 for p in large_model.parameters()) / (1024 * 1024)
    print(f"   ✓ Large model size: ~{large_size:.2f} MB")

    # List all generated files
    print("\n📊 Generated files:")
    for file in sorted(temp_dir.glob("*.pth")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name}: {size_mb:.2f} MB")

    print(f"\n✅ All model files have been generated in: {temp_dir}")
    print("\n💡 To run the benchmarks:")
    print("   cargo bench --bench pytorch_loading")
    print("\n⚠️  Note: The benchmark will look for files in the above directory.")

if __name__ == "__main__":
    main()