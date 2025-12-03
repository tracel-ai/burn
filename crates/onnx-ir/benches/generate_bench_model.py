#!/usr/bin/env python3
# /// script
# python = "3.11"
# dependencies = [
#   "onnx>=1.14",
#   "huggingface_hub",
# ]
# ///
"""Download real ONNX models for benchmarking mmap loading.

This script downloads models from Hugging Face and converts them
to opset 16 for compatibility with onnx-ir.

Usage:
    uv run benches/generate_bench_model.py
"""

import os
import tempfile
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download
from onnx import version_converter, shape_inference


def convert_to_opset16(input_path: Path, output_path: Path) -> None:
    """Convert an ONNX model to opset 16."""
    print(f"  Loading model...")
    model = onnx.load(str(input_path))

    current_opset = model.opset_import[0].version if model.opset_import else 0
    print(f"  Current opset: {current_opset}")

    if current_opset < 16:
        print(f"  Upgrading opset from {current_opset} to 16...")
        converted = version_converter.convert_version(model, 16)
        print(f"  Applying shape inference...")
        converted = shape_inference.infer_shapes(converted)
        onnx.save(converted, str(output_path))
        print(f"  Saved: {output_path}")
    else:
        onnx.save(model, str(output_path))
        print(f"  Already opset {current_opset}, saved as: {output_path}")


def download_minilm(output_dir: Path) -> Path:
    """Download all-MiniLM-L6-v2 (~86MB sentence transformer)."""
    output_path = output_dir / "all-minilm-l6-v2_opset16.onnx"

    if output_path.exists():
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {output_path} ({size:.1f} MB)")
        return output_path

    print("Downloading all-MiniLM-L6-v2 from Hugging Face...")
    original_path = output_dir / "all-minilm-l6-v2.onnx"

    if not original_path.exists():
        downloaded = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model.onnx",
            cache_dir=str(output_dir / "cache"),
        )
        # Copy to our location
        import shutil
        shutil.copy(downloaded, original_path)
        print(f"  Downloaded: {original_path}")

    convert_to_opset16(original_path, output_path)

    size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Final size: {size:.1f} MB")
    return output_path


def download_clip_vision(output_dir: Path) -> Path:
    """Download CLIP ViT-B-32 vision encoder (~336MB)."""
    output_path = output_dir / "clip-vit-b-32-vision_opset16.onnx"

    if output_path.exists():
        size = output_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {output_path} ({size:.1f} MB)")
        return output_path

    print("Downloading CLIP ViT-B-32-vision from Hugging Face...")
    original_path = output_dir / "clip-vit-b-32-vision.onnx"

    if not original_path.exists():
        downloaded = hf_hub_download(
            repo_id="Xenova/clip-vit-base-patch32",
            filename="onnx/vision_model.onnx",
            cache_dir=str(output_dir / "cache"),
        )
        import shutil
        shutil.copy(downloaded, original_path)
        print(f"  Downloaded: {original_path}")

    convert_to_opset16(original_path, output_path)

    size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Final size: {size:.1f} MB")
    return output_path


def main():
    models_dir = Path(tempfile.gettempdir()) / "onnx_ir_bench_models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ONNX Benchmark Model Downloader")
    print("=" * 60)
    print()
    print(f"Output directory: {models_dir}")
    print()

    # Download models
    print("1. Downloading MiniLM (~86 MB)...")
    minilm_path = download_minilm(models_dir)
    print()

    print("2. Downloading CLIP Vision (~336 MB)...")
    clip_path = download_clip_vision(models_dir)
    print()

    print("=" * 60)
    print("Downloads complete!")
    print("=" * 60)
    print()
    print("Models:")
    print(f"  MiniLM: {minilm_path}")
    print(f"  CLIP:   {clip_path}")
    print()
    print("Run benchmarks with:")
    print("  cargo bench --bench mmap_loading")


if __name__ == "__main__":
    main()
