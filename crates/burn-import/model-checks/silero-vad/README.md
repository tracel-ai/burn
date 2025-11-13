# Silero VAD Model Check

This model check verifies that burn-import can correctly handle the Silero VAD (Voice Activity
Detection) model.

## Current Status

⚠️ **Partially Working**: The model demonstrates successful parsing and processing of If/Loop/Scan
operators, but currently fails due to an unsupported Pad operator mode.

### Current Blocker

- ❌ Pad operator with "reflect" mode (only "constant" mode is currently supported)
- LSTM operator
- Not operator

This model serves as an integration test for subgraph and LSTM support.

## Model Information

- **Model**: Silero VAD
- **Source**: https://github.com/snakers4/silero-vad
- **Purpose**: Voice Activity Detection
- **Key Features**: Uses If and LSTM operators for stateful processing

This makes it an excellent test case for burn-import's subgraph support and demonstrates where
future improvements are needed.

## Setup and Run

### Step 1: Download the Model

```bash
# Using Python
python get_model.py

# Or using uv
uv run get_model.py
```

This downloads the Silero VAD ONNX model (~351 KB) to the `artifacts/` directory and extracts node information:
- `artifacts/silero_vad.onnx` - The ONNX model file
- `artifacts/node_info.json` - Detailed analysis of all nodes, operators, and configurations

### Step 2: Build and Test

```bash
# Attempt to build the model (will currently fail due to Pad operator)
cargo build
```

## Backend Support

This model check supports multiple backends:

```bash
# NdArray backend (default, CPU)
cargo run

# LibTorch backend (CPU/CUDA)
cargo run --features tch --no-default-features

# WGPU backend (GPU via WebGPU)
cargo run --features wgpu --no-default-features

# Metal backend (Apple Silicon GPU)
cargo run --features metal --no-default-features
```
