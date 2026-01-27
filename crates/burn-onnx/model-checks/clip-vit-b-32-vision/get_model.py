#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime>=1.22.0",
#   "huggingface-hub>=0.20.0",
#   "numpy",
#   "torch",
# ]
# ///

import os
import sys
import onnx
from onnx import shape_inference, version_converter
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_clip_model(output_path):
    """Download CLIP ViT-B-32-vision model from Hugging Face."""
    print("Downloading CLIP ViT-B-32-vision model from Hugging Face...")

    # Download the ONNX model from Hugging Face
    model_path = hf_hub_download(
        repo_id="Qdrant/clip-ViT-B-32-vision",
        filename="model.onnx",
        cache_dir="./artifacts/cache",
    )

    # Copy to artifacts
    import shutil

    shutil.copy(model_path, output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download ONNX file to {output_path}")

    print(f"✓ Model downloaded to: {output_path}")


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


def get_input_info(model):
    """Extract input information from ONNX model."""
    inputs = []
    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                # Use proper defaults for CLIP vision model
                if "pixel_values" in input_info.name:
                    # CLIP vision uses [batch, channels, height, width]
                    if len(shape) == 0:
                        shape.append(1)  # batch
                    elif len(shape) == 1:
                        shape.append(3)  # channels
                    elif len(shape) == 2:
                        shape.append(224)  # height
                    elif len(shape) == 3:
                        shape.append(224)  # width
                else:
                    shape.append(1)  # Default to 1 for other dynamic dimensions
        inputs.append(
            {
                "name": input_info.name,
                "shape": shape,
                "dtype": input_info.type.tensor_type.elem_type,
            }
        )
    return inputs


def generate_test_data(model_path, output_dir):
    """Generate test input/output data and save as PyTorch tensors."""
    import torch
    import onnxruntime as ort

    print("\nGenerating test data...")

    # Load model to get input shapes
    model = onnx.load(model_path)
    input_infos = get_input_info(model)

    print(f"  Model has {len(input_infos)} inputs:")
    for info in input_infos:
        print(f"    - {info['name']}: shape={info['shape']}, dtype={info['dtype']}")

    # Create reproducible test inputs
    np.random.seed(42)
    test_inputs = {}

    for info in input_infos:
        if info["dtype"] == onnx.TensorProto.INT64:
            # For INT64 inputs, use random integers
            test_input = np.random.randint(0, 1000, size=info["shape"], dtype=np.int64)
        else:
            # For float inputs (like pixel_values), use random floats
            test_input = np.random.rand(*info["shape"]).astype(np.float32)
        test_inputs[info["name"]] = test_input

    # Run inference to get output
    session = ort.InferenceSession(model_path)
    outputs = session.run(None, test_inputs)

    # Save in a format that's easier to load in Rust
    # For CLIP vision, we expect:
    # - Inputs: pixel_values
    # - Outputs: image_embeds (2D)

    # Create a more structured format for Rust
    test_data = {
        "pixel_values": torch.from_numpy(
            test_inputs.get("pixel_values", list(test_inputs.values())[0])
        ),
        "image_embeds": torch.from_numpy(outputs[0]),
    }

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)

    print(f"  ✓ Test data saved to: {test_data_path}")
    print(f"    Input shapes:")
    print(f"      pixel_values: {test_data['pixel_values'].shape}")
    print(f"    Output shapes:")
    print(f"      image_embeds: {test_data['image_embeds'].shape}")


def save_model_info(model_path, output_dir):
    """Save model structure information to a text file."""
    print("\nSaving model information...")

    model = onnx.load(model_path)

    info_path = Path(output_dir) / "model-python.txt"
    with open(info_path, "w") as f:
        f.write("CLIP ViT-B-32-vision Model Information\n")
        f.write("=" * 60 + "\n\n")

        # Input information
        f.write("Inputs:\n")
        for input_info in model.graph.input:
            f.write(f"  - {input_info.name}\n")
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(input_info.type.tensor_type.elem_type)}\n"
            )

        # Output information
        f.write("\nOutputs:\n")
        for output_info in model.graph.output:
            f.write(f"  - {output_info.name}\n")
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(output_info.type.tensor_type.elem_type)}\n"
            )

        # Model statistics
        f.write(f"\nModel Statistics:\n")
        f.write(f"  Opset version: {model.opset_import[0].version}\n")
        f.write(f"  Number of nodes: {len(model.graph.node)}\n")
        f.write(f"  Number of initializers: {len(model.graph.initializer)}\n")

        # Node types summary
        node_types = {}
        for node in model.graph.node:
            op_type = node.op_type
            node_types[op_type] = node_types.get(op_type, 0) + 1

        f.write(f"\nNode types:\n")
        for op_type, count in sorted(node_types.items()):
            f.write(f"  {op_type}: {count}\n")

    print(f"  ✓ Model info saved to: {info_path}")


def main():
    print("=" * 60)
    print("CLIP ViT-B-32-vision Model Preparation Tool")
    print("=" * 60)

    # Setup paths
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    original_path = artifacts_dir / "clip-vit-b-32-vision.onnx"
    processed_path = artifacts_dir / "clip-vit-b-32-vision_opset16.onnx"
    test_data_path = artifacts_dir / "test_data.pt"
    model_info_path = artifacts_dir / "model-python.txt"

    # Check if we already have everything
    if processed_path.exists() and test_data_path.exists() and model_info_path.exists():
        print(f"\n✓ All files already exist:")
        print(f"  Model: {processed_path}")
        print(f"  Test data: {test_data_path}")
        print(f"  Model info: {model_info_path}")
        print("\nNothing to do!")
        return

    # Download model if needed
    if not original_path.exists() and not processed_path.exists():
        print("\nStep 1: Downloading CLIP model...")
        download_clip_model(original_path)

    # Process model if needed
    if not processed_path.exists():
        print("\nStep 2: Processing model...")
        process_model(original_path, processed_path, target_opset=16)

        # Clean up original if we have the processed version
        if original_path.exists() and processed_path.exists():
            original_path.unlink()

    # Generate test data if needed
    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(processed_path, artifacts_dir)

    # Save model info if needed
    if not model_info_path.exists():
        print("\nStep 4: Saving model information...")
        save_model_info(processed_path, artifacts_dir)

    print("\n" + "=" * 60)
    print("✓ CLIP model preparation completed!")
    print(f"  Model: {processed_path}")
    print(f"  Test data: {test_data_path}")
    print(f"  Model info: {model_info_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
