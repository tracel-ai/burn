#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "torch",
#     "torchvision",
# ]
# ///
"""
Download ResNet18 PyTorch model for benchmarking.
This script downloads a pre-trained ResNet18 model from PyTorch Hub
and saves it in a format suitable for benchmarking.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torchvision.models as models

def download_resnet18():
    """Download ResNet18 model and save to temp directory."""

    # Create a temporary directory for the model
    temp_dir = Path(tempfile.gettempdir()) / "burn_resnet18_benchmark"
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = temp_dir / "resnet18.pth"

    # Check if already downloaded
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ ResNet18 already exists at: {output_path}")
        print(f"   Size: {file_size_mb:.1f} MB")
        return str(output_path)

    print("📥 Downloading ResNet18 model...")

    try:
        # Download pre-trained ResNet18 model
        model = models.resnet18(pretrained=True)

        # Save the model state dict (this is what burn-store reads)
        # Using the legacy format for compatibility
        torch.save(model.state_dict(), output_path, _use_new_zipfile_serialization=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully downloaded ResNet18 to: {output_path}")
        print(f"   Size: {file_size_mb:.1f} MB")
        print(f"   Format: PyTorch legacy format")

        # Verify it's readable
        state_dict = torch.load(output_path, map_location='cpu')
        print(f"   Tensors: {len(state_dict)} tensors")

        # Print a few tensor names and shapes for verification
        print("\n   Sample tensors:")
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i < 3:
                print(f"     - {name}: {list(tensor.shape)}")

        return str(output_path)

    except Exception as e:
        print(f"❌ Failed to download ResNet18: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    path = download_resnet18()

    # Write the path to a file that the benchmark can read
    bench_config = Path(tempfile.gettempdir()) / "burn_resnet18_benchmark" / "path.txt"
    bench_config.write_text(path)

    print(f"\n💡 Model ready for benchmarking")
    print(f"   Run: cargo bench --bench resnet18_loading")

if __name__ == "__main__":
    main()