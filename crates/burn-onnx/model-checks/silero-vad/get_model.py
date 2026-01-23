#!/usr/bin/env python3
"""
Download and prepare the Silero VAD model for testing.

This script downloads the Silero VAD ONNX model (opset 18, if-less version) and prepares
it for use with burn-onnx. This version has only 1 If node (for sample rate selection)
making it compatible with static type inference.

See: https://github.com/snakers4/silero-vad/issues/728 for compatibility discussion.
"""

import json
import struct
import urllib.request
import wave
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import onnx
except ImportError:
    print("Error: onnx package not found. Please install it with:")
    print("  pip install onnx")
    exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime package not found. Please install it with:")
    print("  pip install onnxruntime")
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


def download_test_data():
    """Download test audio file from silero-vad repository."""

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    test_wav_path = artifacts_dir / "test.wav"

    if test_wav_path.exists():
        print(f"✓ Test audio already exists at {test_wav_path}")
        return test_wav_path

    # Download the test.wav file from silero-vad tests
    test_url = "https://github.com/snakers4/silero-vad/raw/refs/heads/master/tests/data/test.wav"

    print(f"Downloading test audio from:")
    print(f"  {test_url}")
    print(f"Saving to: {test_wav_path}")

    try:
        urllib.request.urlretrieve(test_url, test_wav_path)
        file_size = test_wav_path.stat().st_size / 1024
        print(f"✓ Download complete! File size: {file_size:.1f} KB")
    except Exception as e:
        print(f"✗ Error downloading test audio: {e}")
        raise

    return test_wav_path


def load_wav(wav_path):
    """Load a WAV file and return audio samples as float32 array normalized to [-1, 1]."""
    with wave.open(str(wav_path), 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Read raw bytes
        raw_data = wav_file.readframes(n_frames)

        # Convert to numpy array based on sample width
        if sample_width == 2:  # 16-bit
            samples = np.frombuffer(raw_data, dtype=np.int16)
            # Normalize to [-1, 1]
            samples = samples.astype(np.float32) / 32768.0
        elif sample_width == 4:  # 32-bit
            samples = np.frombuffer(raw_data, dtype=np.int32)
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # If stereo, convert to mono by averaging channels
        if n_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        return samples, frame_rate


def generate_reference_outputs(model_path, test_wav_path, artifacts_dir):
    """Generate reference outputs using ONNX Runtime for testing."""

    print(f"  Loading test audio from {test_wav_path}...")
    audio_samples, sample_rate = load_wav(test_wav_path)
    print(f"    Sample rate: {sample_rate} Hz")
    print(f"    Audio length: {len(audio_samples)} samples ({len(audio_samples)/sample_rate:.2f} seconds)")

    # Create ONNX Runtime session
    print(f"  Creating ONNX Runtime session...")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    # Silero VAD parameters
    # For 16kHz: chunk_size = 512 (32ms)
    # For 8kHz: chunk_size = 256 (32ms)
    chunk_size = 512 if sample_rate == 16000 else 256
    batch_size = 1

    # Initialize state
    state = np.zeros((2, batch_size, 128), dtype=np.float32)

    # Process audio in chunks and collect outputs
    results = {
        "sample_rate": sample_rate,
        "chunk_size": chunk_size,
        "audio_length_samples": len(audio_samples),
        "test_cases": []
    }

    # Test Case 1: First few chunks from the beginning of the audio
    print(f"  Running inference on test chunks...")
    num_test_chunks = min(10, len(audio_samples) // chunk_size)

    state = np.zeros((2, batch_size, 128), dtype=np.float32)
    for i in range(num_test_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        chunk = audio_samples[start_idx:end_idx]
        if len(chunk) < chunk_size:
            # Pad with zeros if needed
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

        # Prepare inputs
        input_tensor = chunk.reshape(1, -1).astype(np.float32)
        sr_tensor = np.array(sample_rate, dtype=np.int64)

        # Run inference
        outputs = session.run(None, {
            'input': input_tensor,
            'sr': sr_tensor,
            'state': state
        })

        output_prob = float(outputs[0].flatten()[0])
        state = outputs[1]

        results["test_cases"].append({
            "test_name": f"chunk_{i}",
            "chunk_index": i,
            "start_sample": start_idx,
            "input_samples": chunk.tolist(),
            "expected_output": output_prob,
            "state_after": state.tolist()
        })

    # Test Case 2: Random input (for reproducibility test)
    print(f"  Generating random input test case...")
    np.random.seed(42)
    random_input = np.random.randn(chunk_size).astype(np.float32) * 0.1
    state = np.zeros((2, batch_size, 128), dtype=np.float32)

    outputs = session.run(None, {
        'input': random_input.reshape(1, -1),
        'sr': np.array(sample_rate, dtype=np.int64),
        'state': state
    })

    results["test_cases"].append({
        "test_name": "random_seed_42",
        "chunk_index": -1,
        "start_sample": -1,
        "input_samples": random_input.tolist(),
        "expected_output": float(outputs[0].flatten()[0]),
        "state_after": outputs[1].tolist()
    })

    # Test Case 3: Zero input (silence)
    print(f"  Generating silence test case...")
    zero_input = np.zeros(chunk_size, dtype=np.float32)
    state = np.zeros((2, batch_size, 128), dtype=np.float32)

    outputs = session.run(None, {
        'input': zero_input.reshape(1, -1),
        'sr': np.array(sample_rate, dtype=np.int64),
        'state': state
    })

    results["test_cases"].append({
        "test_name": "silence",
        "chunk_index": -1,
        "start_sample": -1,
        "input_samples": zero_input.tolist(),
        "expected_output": float(outputs[0].flatten()[0]),
        "state_after": outputs[1].tolist()
    })

    # Save results
    output_path = artifacts_dir / "reference_outputs.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Reference outputs saved to {output_path}")
    print(f"    Total test cases: {len(results['test_cases'])}")


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

    # Download test data
    print("Downloading test data...")
    try:
        test_wav_path = download_test_data()
    except Exception as e:
        print(f"✗ Error downloading test data: {e}")
        raise

    print()

    # Generate reference outputs
    print("Generating reference outputs...")
    try:
        generate_reference_outputs(model_path, test_wav_path, artifacts_dir)
    except Exception as e:
        print(f"✗ Error generating reference outputs: {e}")
        raise

    print()
    print("="*80)
    print("Model preparation complete!")
    print("="*80)
    print()
    print("Silero VAD (opset 18, if-less) has only 1 If node for sample rate selection.")
    print("This makes it compatible with burn-onnx's static type inference.")
    print()
    print("See: https://github.com/snakers4/silero-vad/issues/728")
    print()
    print("Generated files:")
    print(f"  - {model_path} (ONNX model)")
    print(f"  - {artifacts_dir / 'node_info.json'} (node analysis)")
    print(f"  - {test_wav_path} (test audio)")
    print(f"  - {artifacts_dir / 'reference_outputs.json'} (reference test outputs)")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test: cargo run")
    print()


if __name__ == "__main__":
    download_model()
