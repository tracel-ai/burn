#!/usr/bin/env python3
"""
Generate ONNX model that tests outer-scope variable references in Scan subgraphs.

Pattern:
    y = Relu(x)           # Parent computes y (shape [3])
    z = Scan(sequence) {
        body: scan_out = Add(elem, y)  # References y from parent (not a scan input!)
    }

The scan body references 'y' WITHOUT declaring it as an input.
Each element of the sequence is added to y.
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime not available, skipping runtime validation")


def build_model():
    """
    Build Scan model with outer-scope reference.

    Structure:
        y = Relu(x)                    # Parent computes y (1D tensor)
        z = Scan(sequence) {
            body: scan_out = Add(elem, y)  # References y from parent!
        }

    The scan body references 'y' from the parent scope in scan_out computation.
    """
    # Scan body: scan_out = elem + y (references y from outer scope!)
    body_add = helper.make_node('Add', inputs=['elem', 'y'], outputs=['scan_out'])

    body_graph = helper.make_graph(
        nodes=[body_add],
        name='scan_body',
        inputs=[
            helper.make_tensor_value_info('elem', TensorProto.FLOAT, [3]),   # Scan input element
            # Note: 'y' is NOT declared as input - it's an outer-scope reference
        ],
        outputs=[
            helper.make_tensor_value_info('scan_out', TensorProto.FLOAT, [3]),   # Scan output
        ],
    )

    # Main graph
    relu_node = helper.make_node('Relu', inputs=['x'], outputs=['y'])

    scan_node = helper.make_node(
        'Scan',
        inputs=['sequence'],
        outputs=['scan_output'],
        body=body_graph,
        num_scan_inputs=1,
    )

    main_graph = helper.make_graph(
        nodes=[relu_node, scan_node],
        name='outer_scope_scan',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [3]),  # y will be [3]
            helper.make_tensor_value_info('sequence', TensorProto.FLOAT, [4, 3]),  # 4 elements of shape [3]
        ],
        outputs=[
            helper.make_tensor_value_info('scan_output', TensorProto.FLOAT, [4, 3]),
        ],
    )

    model = helper.make_model(main_graph, producer_name='burn-onnx-test',
                               opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def test_model(model):
    """Test the model with ONNX Runtime."""
    if not HAS_ORT:
        print("Skipping runtime test (onnxruntime not available)")
        return

    x = np.array([-1.0, 2.0, 0.5], dtype=np.float32)
    sequence = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ], dtype=np.float32)

    sess = ort.InferenceSession(model.SerializeToString())

    # y = Relu(x) = [0, 2, 0.5]
    y = np.maximum(x, 0)

    out = sess.run(None, {'x': x, 'sequence': sequence})[0]

    # scan_out[i] = sequence[i] + y
    expected = sequence + y

    print("=== Scan Outer Scope Reference Test ===")
    print(f"x: {x}")
    print(f"y = Relu(x): {y}")
    print(f"sequence:\n{sequence}")
    print(f"Output (each elem + y):\n{out}")
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(out, expected)}")

    assert np.allclose(out, expected), "Scan output mismatch!"
    print("\nTest passed!")


def main():
    model = build_model()
    onnx.save(model, 'outer_scope_scan.onnx')
    print("Saved outer_scope_scan.onnx")

    test_model(model)


if __name__ == '__main__':
    main()
