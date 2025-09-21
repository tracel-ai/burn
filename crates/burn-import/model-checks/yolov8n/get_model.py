#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx>=1.17.0",
#   "onnxruntime>=1.18.0",
#   "ultralytics>=8.3.0",
#   "numpy",
#   "pillow",
#   "torch",
# ]
# ///

import os
import sys
import onnx
from onnx import shape_inference, version_converter
import numpy as np
from pathlib import Path


def get_input_shape(model):
    """Extract input shape from ONNX model."""
    input_info = model.graph.input[0]
    shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        else:
            shape.append(1)  # Default to 1 for dynamic dimensions

    # Ensure valid YOLO input shape
    if len(shape) != 4 or shape[2] == 0 or shape[2] > 2000:
        return [1, 3, 640, 640]
    return shape


def download_and_convert_model(output_path):
    """Download YOLOv8n model and export to ONNX format."""
    from ultralytics import YOLO

    print("Downloading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    print("Exporting to ONNX format...")
    model.export(format="onnx", simplify=True)

    # Move exported file to artifacts
    exported_file = Path("yolov8n.onnx")
    if exported_file.exists():
        exported_file.rename(output_path)

    # Clean up PyTorch file
    pt_file = Path("yolov8n.pt")
    if pt_file.exists():
        pt_file.unlink()

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create ONNX file at {output_path}")


def process_model(input_path, output_path, target_opset=16):
    """Load, upgrade opset, and apply shape inference to model."""
    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    # Check and upgrade opset if needed
    current_opset = model.opset_import[0].version
    if current_opset < target_opset:
        print(f"Upgrading opset from {current_opset} to {target_opset}...")
        model = version_converter.convert_version(model, target_opset)

    # Apply shape inference
    print("Applying shape inference...")
    model = shape_inference.infer_shapes(model)

    # Save processed model
    onnx.save(model, output_path)
    print(f"✓ Processed model saved to: {output_path}")

    return model


def generate_test_data(model_path, output_dir):
    """Generate test input/output data and save as PyTorch tensors."""
    import torch
    import onnxruntime as ort

    print("\nGenerating test data...")

    # Load model to get input shape
    model = onnx.load(model_path)
    input_shape = get_input_shape(model)
    print(f"  Input shape: {input_shape}")

    # Create reproducible test input
    np.random.seed(42)
    test_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference to get output
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: test_input})

    # Save as PyTorch tensors
    test_data = {
        'input': torch.from_numpy(test_input),
        'output': torch.from_numpy(outputs[0])
    }

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)

    print(f"  ✓ Test data saved to: {test_data_path}")
    print(f"    Input shape: {test_input.shape}, Output shape: {outputs[0].shape}")


def main():
    print("=" * 60)
    print("YOLOv8n Model Preparation Tool")
    print("=" * 60)

    # Setup paths
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    original_path = artifacts_dir / "yolov8n.onnx"
    processed_path = artifacts_dir / "yolov8n_opset16.onnx"
    test_data_path = artifacts_dir / "test_data.pt"

    # Check if we already have everything
    if processed_path.exists() and test_data_path.exists():
        print(f"\n✓ All files already exist:")
        print(f"  Model: {processed_path}")
        print(f"  Test data: {test_data_path}")
        print("\nNothing to do!")
        return

    # Download and convert if needed
    if not original_path.exists() and not processed_path.exists():
        print("\nStep 1: Downloading and converting YOLOv8n model...")
        download_and_convert_model(original_path)

    # Process model if needed
    if not processed_path.exists():
        print("\nStep 2: Processing model...")
        process_model(original_path, processed_path, target_opset=16)

        # Clean up original if we have the processed version
        if original_path.exists():
            original_path.unlink()

    # Generate test data if needed
    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(processed_path, artifacts_dir)

    print("\n" + "=" * 60)
    print("✓ YOLOv8n model preparation completed!")
    print(f"  Model: {processed_path}")
    print(f"  Test data: {test_data_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)