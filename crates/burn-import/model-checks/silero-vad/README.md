# Silero VAD Model Check

This model check verifies that burn-import can correctly handle the Silero VAD (Voice Activity Detection) model, which uses advanced ONNX operators including If, Loop, and Scan.

## Current Status

⚠️ **Partially Working**: The model demonstrates successful parsing and processing of If/Loop/Scan operators, but currently fails due to an unsupported Pad operator mode.

### What Works
- ✅ If operator parsing and subgraph processing
- ✅ Loop operator with nested subgraphs
- ✅ Scan operator integration
- ✅ Complex nested control flow

### Current Blocker
- ❌ Pad operator with "reflect" mode (only "constant" mode is currently supported)

This model serves as an integration test for subgraph support and a target for future Pad operator improvements.

## Model Information

- **Model**: Silero VAD
- **Source**: https://github.com/snakers4/silero-vad
- **Purpose**: Voice Activity Detection
- **Key Features**: Uses If/Loop/Scan operators for stateful processing

## ONNX Operators Used

The Silero VAD model tests critical ONNX operators:

- **If**: Conditional execution (2 instances in the model) ✅
- **Loop**: Iterative execution with state (1 instance) ✅
- **Scan**: Sequential processing with LSTM layers ✅
- **LSTM**: Long Short-Term Memory layers for audio processing
- **Pad**: Padding operator (currently blocked by "reflect" mode) ❌

This makes it an excellent test case for burn-import's subgraph support and demonstrates where future improvements are needed.

## Setup and Run

### Step 1: Download the Model

```bash
# Using Python
python get_model.py

# Or using uv
uv run get_model.py
```

This downloads the Silero VAD ONNX model (~351 KB) to the `artifacts/` directory.

### Step 2: Build and Test

```bash
# Attempt to build the model (will currently fail due to Pad operator)
cargo build
```

## Current Build Output

Currently, the build process successfully demonstrates If/Loop/Scan operator support but fails at the Pad operator:

```
INFO onnx_ir::pipeline: Parsing ONNX file: artifacts/silero_vad.onnx
DEBUG onnx_ir::pipeline:  PHASE 1: Initialization
DEBUG onnx_ir::pipeline:  PHASE 2: Node Conversion
...
[Successfully processes If/Loop/Scan subgraphs]
...
ERROR: Failed to extract config for node pad1 (type: Pad):
       InvalidAttribute { name: "mode",
       reason: "only constant mode is supported, given mode is reflect" }
```

This demonstrates that:
- ✅ If/Loop/Scan operators are correctly parsed and processed
- ✅ Nested subgraphs work as expected
- ❌ Pad operator needs "reflect" mode support (future enhancement)

## Future Expected Output (Once Pad is Supported)

Once the Pad operator's "reflect" mode is implemented, the test will:
1. Initialize the Silero VAD model
2. Create a sample audio input (512 samples)
3. Run voice activity detection inference
4. Display the voice probability output (0.0 to 1.0)

Example expected output:
```
========================================
Silero VAD Model Test
========================================

This model tests burn-import's support for:
  - If operators (conditional execution)
  - Loop operators (iterative execution)
  - Scan operators (sequential processing)

Initializing Silero VAD model...
  ✓ Model initialized in 123ms

Running model inference...
  ✓ Inference completed in 45ms

Model output:
  Voice probability: 0.1234 (0.0 = no voice, 1.0 = voice)
  ✓ Output is a valid probability

========================================
Model test completed successfully!
========================================

✓ If/Loop/Scan operators are working correctly
✓ Model can be imported and executed with burn
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

## What This Tests

- ✅ ONNX If operator (conditional branching)
- ✅ ONNX Loop operator (stateful iteration)
- ✅ ONNX Scan operator (sequential processing)
- ✅ Nested subgraphs (operators within operators)
- ✅ State management (LSTM hidden and cell states)
- ✅ Scalar condition handling (rank-0 tensors converted to bools)
- ✅ Complex model topology with multiple control flow paths
