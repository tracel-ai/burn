# Silero VAD Model Check

This model check verifies that burn-import can correctly handle the Silero VAD (Voice Activity
Detection) model and produces outputs matching ONNX Runtime.

## Current Status

**Working**: The model is successfully imported and produces outputs matching ONNX Runtime.

## Model Information

- **Model**: Silero VAD (opset 18, if-less version)
- **Source**: https://github.com/snakers4/silero-vad
- **Purpose**: Voice Activity Detection
- **Key Features**: Uses Conv, Gemm, and a single If node for sample rate selection

The if-less version has only 1 If node (for 16kHz vs 8kHz sample rate selection), making it
much simpler than the full model which has 25 If nodes.

See: https://github.com/snakers4/silero-vad/issues/728 for compatibility discussion.

## Setup

### Step 1: Download the Model and Generate Reference Outputs

```bash
# Using Python
python get_model.py

# Or using uv
uv run get_model.py
```

This downloads:
- `artifacts/silero_vad.onnx` - The ONNX model file
- `artifacts/node_info.json` - Detailed analysis of all nodes, operators, and configurations
- `artifacts/test.wav` - Test audio file from silero-vad repository
- `artifacts/reference_outputs.json` - Reference outputs from ONNX Runtime for validation

### Step 2: Build and Run Tests

```bash
cargo build
cargo run
```

The test suite runs 12 test cases:
- 10 audio chunks from the test.wav file
- 1 random input test (reproducible with seed 42)
- 1 silence test (all zeros)

Each test compares the Burn model output against ONNX Runtime reference outputs with a 1% tolerance.

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

## Test Output Example

```
========================================
Silero VAD Model Test Suite
========================================

Loading reference outputs...
  Loaded 12 test cases (sample rate: 16000 Hz)

Initializing Silero VAD model...
  Model initialized

Running test cases...
------------------------------------------------------------
  [PASS] chunk_0: output=0.000589 (expected=0.000589)
  [PASS] chunk_1: output=0.000589 (expected=0.000528)
  ...
  [PASS] silence: output=0.000592 (expected=0.000592)
------------------------------------------------------------

========================================
Test Summary
========================================
  Total tests: 12
  Passed: 12
  Failed: 0

All tests passed!
The Burn model produces outputs matching ONNX Runtime.
```
