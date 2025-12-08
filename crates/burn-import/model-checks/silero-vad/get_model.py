#!/usr/bin/env python3
"""
Download and prepare the Silero VAD model for testing.

This script downloads the Silero VAD ONNX model (opset 18, if-less version) and prepares
it for use with burn-import. This version has only 1 If node (for sample rate selection)
making it compatible with static type inference.

See: https://github.com/snakers4/silero-vad/issues/728 for compatibility discussion.
"""

import json
import urllib.request
from pathlib import Path
from collections import defaultdict

try:
    import onnx
except ImportError:
    print("Error: onnx package not found. Please install it with:")
    print("  pip install onnx")
    exit(1)


def extract_node_info(model_path, artifacts_dir):
    """Extract node types and configurations from the ONNX model."""
    print("Extracting node information from ONNX model...")

    # Load the ONNX model (without external data since we only need structure)
    model = onnx.load(str(model_path), load_external_data=False)

    # Check for external data
    external_files = set()
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for ext_data in init.external_data:
                if ext_data.key == 'location':
                    external_files.add(ext_data.value)

    if external_files:
        print(f"⚠️  Model requires external data files: {external_files}")
        print("   These files are missing from the repository!")

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
        "external_data_files": list(external_files),
        "nodes": node_details
    }

    # Save to JSON file
    output_path = artifacts_dir / "node_info.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Node information extracted to {output_path}")
    print(f"  Opset version: {summary['opset_version']}")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Unique node types: {len(node_types)}")
    print(f"  Node type distribution:")
    for op_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {op_type}: {count}")

    return summary


def download_model():
    """Download the Silero VAD ONNX model (opset 18, if-less version)."""

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "silero_vad.onnx"

    model_existed = model_path.exists()

    # Skip download if model already exists
    if model_existed:
        print(f"✓ Model already exists at {model_path}")
        print(f"  File size: {model_path.stat().st_size / 1024:.1f} KB")
        print()
    else:
        # Download the opset 18 if-less model
        # Note: This model has external data that is missing from the repo
        model_url = "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad_op18_ifless.onnx"

        print(f"Downloading Silero VAD model (opset 18, if-less) from:")
        print(f"  {model_url}")
        print(f"Saving to: {model_path}")
        print()

        try:
            urllib.request.urlretrieve(model_url, model_path)
            file_size = model_path.stat().st_size / 1024
            print(f"✓ Download complete! File size: {file_size:.1f} KB")
            print()
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            raise

    # Extract node information from the ONNX model
    try:
        extract_node_info(model_path, artifacts_dir)
    except Exception as e:
        print(f"✗ Error extracting node information: {e}")
        raise

    print()
    print("="*80)
    print("Model preparation complete!")
    print("="*80)
    print()
    print("Silero VAD (opset 18, if-less) has only 1 If node for sample rate selection.")
    print("This makes it compatible with burn-import's static type inference.")
    print()
    print("See: https://github.com/snakers4/silero-vad/issues/728")
    print()
    print("Generated files:")
    print(f"  - {model_path} (ONNX model)")
    print(f"  - {artifacts_dir / 'node_info.json'} (node analysis)")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test: cargo run")
    print()


if __name__ == "__main__":
    download_model()
