#!/usr/bin/env python3
"""
Download and prepare the Silero VAD model for testing.

This script downloads the Silero VAD ONNX model and prepares it for use with burn-import.
The model uses If/Loop/Scan operators which test the subgraph support.
"""

import urllib.request
from pathlib import Path


def download_model():
    """Download the Silero VAD ONNX model."""

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "silero_vad.onnx"

    # Skip download if model already exists
    if model_path.exists():
        print(f"✓ Model already exists at {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
        return

    # Download the model
    model_url = "https://github.com/snakers4/silero-vad/raw/9623ce72da2eb2f08466d67ddda11f5636486172/src/silero_vad/data/silero_vad.onnx"

    print(f"Downloading Silero VAD model from:")
    print(f"  {model_url}")
    print(f"Saving to: {model_path}")
    print()

    try:
        urllib.request.urlretrieve(model_url, model_path)
        file_size = model_path.stat().st_size / 1024
        print(f"✓ Download complete! File size: {file_size:.1f} KB")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise

    print()
    print("="*80)
    print("Model download complete!")
    print("="*80)
    print()
    print("The Silero VAD model uses advanced ONNX operators:")
    print("  - If: Conditional execution (2 instances)")
    print("  - Loop: Iterative execution (1 instance)")
    print("  - Scan: Sequential processing")
    print()
    print("These operators test burn-import's subgraph support.")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test: cargo run")
    print()


if __name__ == "__main__":
    download_model()
