# Silero VAD Model Check

This model check verifies that burn-import can correctly handle the Silero VAD (Voice Activity
Detection) model.

## Current Status

⚠️ **In Progress**: The model downloads successfully but encounters issues during code generation:

1. **Fixed**: Reshape operator now handles Scalar inputs
2. **Current Blocker**: Constant lifting in If subgraphs - constants from the parent graph
   are not accessible when lifting constants inside If branches

The silero_vad_op18_ifless.onnx model has only 1 If node but the branches reference
constants from the parent scope that aren't properly resolved.

## Model Information

- **Model**: Silero VAD (opset 18, if-less version)
- **Source**: https://github.com/snakers4/silero-vad
- **Purpose**: Voice Activity Detection
- **Key Features**: Uses Conv, Gemm, and a single If node for sample rate selection

The if-less version has only 1 If node (for 16kHz vs 8kHz sample rate selection), making it
much simpler than the full model which has 25 If nodes.

See: https://github.com/snakers4/silero-vad/issues/728 for compatibility discussion.

## Setup

### Step 1: Download the Model

```bash
# Using Python
python get_model.py

# Or using uv
uv run get_model.py
```

This downloads the Silero VAD ONNX model to the `artifacts/` directory and extracts node information:
- `artifacts/silero_vad.onnx` - The ONNX model file
- `artifacts/node_info.json` - Detailed analysis of all nodes, operators, and configurations

### Step 2: Build (currently fails)

```bash
cargo build
```

## Backend Support

Once the issues are resolved, this model check will support multiple backends:

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
