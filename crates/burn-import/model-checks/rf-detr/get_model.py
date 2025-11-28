#!/usr/bin/env -S uv run --python 3.11 --script

# /// script
# python = "3.11"
# dependencies = [
#   "torch==2.6.*",
#   "torchvision==0.21.*",
#   "onnx",
#   "onnxruntime",
#   "numpy",
#   "rfdetr[onnxexport]",
# ]
# ///

"""
Download and prepare the RF-DETR model for testing with burn-import.

RF-DETR (Roboflow DETR) is a real-time object detection model that combines
the DETR (Detection Transformer) architecture with optimizations for speed.

This script exports the RFDETRSmall model to ONNX format for testing burn-import's
ability to handle transformer-based detection models.

Related issue: https://github.com/tracel-ai/burn/issues/4052
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx


def get_input_shape(model):
    """Extract input shape from ONNX model."""
    input_info = model.graph.input[0]
    shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        else:
            shape.append(1)  # Default to 1 for dynamic dimensions

    # Ensure valid RF-DETR input shape (batch, channels, height, width)
    # RF-DETR uses 560x560 by default
    if len(shape) != 4 or shape[2] == 0 or shape[2] > 2000:
        return [1, 3, 560, 560]
    return shape


def extract_node_info(model_path, artifacts_dir):
    """Extract node types and configurations from the ONNX model."""
    print("Extracting node information from ONNX model...")

    # Load the ONNX model
    model = onnx.load(str(model_path))

    # Collect node information
    node_types = defaultdict(int)
    node_details = []

    def process_graph(graph, graph_name="main"):
        """Recursively process a graph and its subgraphs."""
        for idx, node in enumerate(graph.node):
            node_types[node.op_type] += 1

            # Extract node details
            node_info = {
                "graph": graph_name,
                "index": idx,
                "op_type": node.op_type,
                "name": node.name if node.name else f"{node.op_type}_{idx}",
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {}
            }

            # Extract attributes
            for attr in node.attribute:
                attr_name = attr.name
                # Get attribute value based on type
                if attr.HasField('f'):
                    node_info["attributes"][attr_name] = float(attr.f)
                elif attr.HasField('i'):
                    node_info["attributes"][attr_name] = int(attr.i)
                elif attr.HasField('s'):
                    node_info["attributes"][attr_name] = attr.s.decode('utf-8') if attr.s else ""
                elif attr.HasField('t'):
                    node_info["attributes"][attr_name] = "<tensor>"
                elif attr.floats:
                    node_info["attributes"][attr_name] = list(attr.floats)
                elif attr.ints:
                    node_info["attributes"][attr_name] = list(attr.ints)
                elif attr.strings:
                    node_info["attributes"][attr_name] = [s.decode('utf-8') for s in attr.strings]
                elif attr.HasField('g'):
                    # Subgraph - recursively process it
                    subgraph_name = f"{graph_name}.{node.op_type}_{idx}.{attr_name}"
                    node_info["attributes"][attr_name] = f"<subgraph: {subgraph_name}>"
                    process_graph(attr.g, subgraph_name)
                elif attr.graphs:
                    subgraph_names = []
                    for g_idx, subgraph in enumerate(attr.graphs):
                        subgraph_name = f"{graph_name}.{node.op_type}_{idx}.{attr_name}_{g_idx}"
                        subgraph_names.append(subgraph_name)
                        process_graph(subgraph, subgraph_name)
                    node_info["attributes"][attr_name] = f"<subgraphs: {', '.join(subgraph_names)}>"
                else:
                    node_info["attributes"][attr_name] = "<unknown>"

            node_details.append(node_info)

    # Process the main graph
    process_graph(model.graph, "main")

    # Create summary
    summary = {
        "model_name": model.graph.name,
        "opset_version": model.opset_import[0].version if model.opset_import else "unknown",
        "total_nodes": len(node_details),
        "node_type_counts": dict(sorted(node_types.items())),
        "nodes": node_details
    }

    # Save to JSON file
    output_path = artifacts_dir / "node_info.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Node information extracted to {output_path}")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Unique node types: {len(node_types)}")
    print(f"  Node type distribution:")
    for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"    - {op_type}: {count}")
    if len(node_types) > 15:
        print(f"    ... and {len(node_types) - 15} more types")

    return summary


def generate_test_data(model_path, output_path):
    """Generate test input/output data and save as PyTorch tensors."""
    import torch
    import onnxruntime as ort

    print("Generating test data...")

    # Load model to get input shape
    model = onnx.load(str(model_path))
    input_shape = get_input_shape(model)
    print(f"  Input shape: {input_shape}")

    # Create reproducible test input
    np.random.seed(42)
    test_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference to get output
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: test_input})

    # RF-DETR has two outputs: dets (boxes) and labels (class scores)
    # Save as PyTorch tensors
    test_data = {
        'input': torch.from_numpy(test_input),
        'output_dets': torch.from_numpy(outputs[0]),
        'output_labels': torch.from_numpy(outputs[1])
    }

    torch.save(test_data, output_path)

    print(f"  Test data saved to: {output_path}")
    print(f"    Input shape: {test_input.shape}")
    print(f"    Output dets shape: {outputs[0].shape}")
    print(f"    Output labels shape: {outputs[1].shape}")


def download_and_export_model():
    """Download RF-DETR model and export to ONNX format."""
    from rfdetr import RFDETRSmall

    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "rf_detr_small.onnx"
    test_data_path = artifacts_dir / "rf_detr_small_test_data.pt"

    # Check if we already have everything
    if model_path.exists() and test_data_path.exists():
        print(f"All files already exist:")
        print(f"  Model: {model_path}")
        print(f"  Test data: {test_data_path}")
        print("\nTo re-download, delete the artifacts directory and run again.")
        return

    print("=" * 60)
    print("RF-DETR Small Model Preparation Tool")
    print("=" * 60)
    print()

    # Download and export if model doesn't exist
    if not model_path.exists():
        # Download and initialize the model
        print("Step 1: Downloading RF-DETR Small model...")
        model = RFDETRSmall()
        print("  Model downloaded and initialized")

        # Export to ONNX using RF-DETR's built-in export method
        print()
        print("Step 2: Exporting model to ONNX format...")

        # RF-DETR exports to output/inference_model.onnx by default
        exported_path = model.export()
        print(f"  Exported to: {exported_path}")

        # Move the exported file to artifacts directory
        exported_file = Path(exported_path)
        if exported_file.exists():
            shutil.move(str(exported_file), str(model_path))
            # Clean up the output directory if empty
            output_dir = exported_file.parent
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()

        # Clean up any downloaded weights file
        pth_files = list(Path(".").glob("*.pth"))
        for pth_file in pth_files:
            pth_file.unlink()
            print(f"  Cleaned up: {pth_file}")

        print(f"  Model saved to {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Extract node information
        print()
        print("Step 3: Analyzing ONNX model structure...")
        extract_node_info(model_path, artifacts_dir)

    # Generate test data if needed
    if not test_data_path.exists():
        print()
        print("Step 4: Generating test data...")
        generate_test_data(model_path, test_data_path)

    print()
    print("=" * 60)
    print("RF-DETR model preparation completed!")
    print("=" * 60)
    print()
    print("The RF-DETR model is a transformer-based object detector that uses:")
    print("  - Multi-head self-attention layers")
    print("  - Cross-attention for object queries")
    print("  - Deformable attention mechanisms")
    print()
    print("Generated files:")
    print(f"  - {model_path} (ONNX model)")
    print(f"  - {test_data_path} (test input/output data)")
    print(f"  - {artifacts_dir / 'node_info.json'} (node analysis)")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test: cargo run")
    print()
    print("Note: This model is used to test burn-import's handling of")
    print("transformer-based architectures. Related issue:")
    print("  https://github.com/tracel-ai/burn/issues/4052")
    print()


if __name__ == "__main__":
    try:
        download_and_export_model()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
