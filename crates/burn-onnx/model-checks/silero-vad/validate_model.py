#!/usr/bin/env python3
"""
Validate the Silero VAD ONNX model independently.

This script:
1. Checks if the model is valid using onnx.checker
2. Runs inference using ONNX Runtime to verify it works
"""

import numpy as np
from pathlib import Path

try:
    import onnx
    from onnx import checker
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


def validate_model():
    """Validate the Silero VAD ONNX model."""

    model_path = Path("artifacts/silero_vad.onnx")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python get_model.py' first to download the model.")
        return False

    print("=" * 80)
    print("Silero VAD ONNX Model Validation")
    print("=" * 80)
    print()

    # Step 1: Load and check the model structure
    print("Step 1: Loading ONNX model...")
    try:
        model = onnx.load(str(model_path))
        print(f"  ✓ Model loaded successfully")
        print(f"  - Opset version: {model.opset_import[0].version}")
        print(f"  - Graph name: {model.graph.name}")
        print(f"  - Number of nodes: {len(model.graph.node)}")
        print(f"  - Number of initializers: {len(model.graph.initializer)}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    print()

    # Step 2: Validate with ONNX checker
    print("Step 2: Validating model with onnx.checker...")
    try:
        checker.check_model(model, full_check=True)
        print("  ✓ Model passed ONNX validation (full check)")
    except Exception as e:
        print(f"  ✗ Model validation failed: {e}")
        return False
    print()

    # Step 3: Print model inputs/outputs
    print("Step 3: Model inputs and outputs...")
    print("  Inputs:")
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"    - {inp.name}: {dtype} {shape}")

    print("  Outputs:")
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"    - {out.name}: {dtype} {shape}")
    print()

    # Step 4: Run inference with ONNX Runtime
    print("Step 4: Running inference with ONNX Runtime...")
    try:
        # Create inference session
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        print("  ✓ ONNX Runtime session created successfully")

        # Get input details
        input_details = session.get_inputs()
        output_details = session.get_outputs()

        print(f"  - Session inputs: {[i.name for i in input_details]}")
        print(f"  - Session outputs: {[o.name for o in output_details]}")

        # Prepare sample inputs based on model signature
        # Silero VAD expects:
        # - input: audio chunk [batch, samples] - typically 512 samples for 16kHz
        # - sr: sample rate (int64)
        # - h: hidden state [2, batch, 64]
        # - c: cell state [2, batch, 64]

        batch_size = 1
        # Silero VAD chunk sizes:
        # - 16kHz: 512 samples (32ms)
        # - 8kHz: 256 samples (32ms)
        # The model also supports larger chunks: 768, 1024, 1536 for 16kHz
        chunk_size = 512  # 512 samples for 16kHz (32ms)

        # Prepare inputs based on Silero VAD documentation
        # https://github.com/snakers4/silero-vad
        inputs = {
            'input': np.random.randn(batch_size, chunk_size).astype(np.float32),
            'sr': np.array(16000, dtype=np.int64),  # 16kHz sample rate
            'state': np.zeros((2, batch_size, 128), dtype=np.float32),  # LSTM hidden state
        }

        for name, value in inputs.items():
            print(f"  - Input '{name}': shape={value.shape}, dtype={value.dtype}")

        # Run inference
        print()
        print("  Running inference...")
        outputs = session.run(None, inputs)

        print("  ✓ Inference completed successfully!")
        print()
        print("  Output values:")
        for i, (out_detail, out_value) in enumerate(zip(output_details, outputs)):
            print(f"    - {out_detail.name}: shape={out_value.shape}, dtype={out_value.dtype}")
            if out_value.size <= 10:
                print(f"      values: {out_value}")
            else:
                print(f"      sample: {out_value.flat[:5]}...")

    except Exception as e:
        print(f"  ✗ ONNX Runtime inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("✓ All validation checks passed!")
    print("=" * 80)
    print()
    print("The ONNX model is valid and can run inference successfully.")
    print("The issue with burn-onnx is in the code generation, not the model itself.")

    return True


if __name__ == "__main__":
    success = validate_model()
    exit(0 if success else 1)
