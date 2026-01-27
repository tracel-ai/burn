#!/usr/bin/env python3
"""
Generate ONNX model that tests MULTIPLE outer-scope variable references in subgraphs.

This tests that DeferredGraph correctly handles multiple variables from parent scope.

Pattern:
    y1 = Relu(x)            # Parent computes y1
    y2 = Sigmoid(x)         # Parent computes y2
    y3 = Tanh(x)            # Parent computes y3
    z = If(condition) {
        then: Add(Add(y1, y2), y3)   # References y1, y2, y3 from parent
        else: Mul(Mul(y1, y2), y3)   # References y1, y2, y3 from parent
    }

The subgraph references THREE variables (y1, y2, y3) WITHOUT declaring them as inputs.
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
    Build If model with multiple outer-scope references.

    Structure:
        y1 = Relu(x)
        y2 = Sigmoid(x)
        y3 = Tanh(x)
        z = If(condition) {
            then: y1 + y2 + y3
            else: y1 * y2 * y3
        }
    """
    # Then branch: y1 + y2 + y3 (all come from outer scope)
    then_add1 = helper.make_node('Add', inputs=['y1', 'y2'], outputs=['then_sum1'])
    then_add2 = helper.make_node('Add', inputs=['then_sum1', 'y3'], outputs=['then_out'])
    then_graph = helper.make_graph(
        nodes=[then_add1, then_add2],
        name='then_branch',
        inputs=[],  # No explicit inputs - y1, y2, y3 come from outer scope
        outputs=[helper.make_tensor_value_info('then_out', TensorProto.FLOAT, [2, 3])],
    )

    # Else branch: y1 * y2 * y3 (all come from outer scope)
    else_mul1 = helper.make_node('Mul', inputs=['y1', 'y2'], outputs=['else_prod1'])
    else_mul2 = helper.make_node('Mul', inputs=['else_prod1', 'y3'], outputs=['else_out'])
    else_graph = helper.make_graph(
        nodes=[else_mul1, else_mul2],
        name='else_branch',
        inputs=[],  # No explicit inputs - y1, y2, y3 come from outer scope
        outputs=[helper.make_tensor_value_info('else_out', TensorProto.FLOAT, [2, 3])],
    )

    # Main graph
    relu_node = helper.make_node('Relu', inputs=['x'], outputs=['y1'])
    sigmoid_node = helper.make_node('Sigmoid', inputs=['x'], outputs=['y2'])
    tanh_node = helper.make_node('Tanh', inputs=['x'], outputs=['y3'])
    if_node = helper.make_node('If', inputs=['condition'], outputs=['output'],
                                then_branch=then_graph, else_branch=else_graph)

    main_graph = helper.make_graph(
        nodes=[relu_node, sigmoid_node, tanh_node, if_node],
        name='outer_scope_multi_var',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('condition', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])],
    )

    model = helper.make_model(main_graph, producer_name='burn-onnx-test',
                               opset_imports=[helper.make_opsetid("", 16)])
    # Use IR version 8 for compatibility with ONNX Runtime
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_model(model):
    """Test the model with ONNX Runtime."""
    if not HAS_ORT:
        print("Skipping runtime test (onnxruntime not available)")
        return

    x = np.array([[-1.5, 2.0, -0.5], [3.0, -2.0, 1.0]], dtype=np.float32)

    sess = ort.InferenceSession(model.SerializeToString())

    # Compute expected values
    y1 = np.maximum(x, 0)      # Relu
    y2 = sigmoid(x)            # Sigmoid
    y3 = np.tanh(x)            # Tanh

    out_then = sess.run(None, {'x': x, 'condition': np.array(True, dtype=bool)})[0]
    out_else = sess.run(None, {'x': x, 'condition': np.array(False, dtype=bool)})[0]

    expected_then = y1 + y2 + y3
    expected_else = y1 * y2 * y3

    print("=== Multi-Variable Outer Scope Reference Test ===")
    print(f"x:\n{x}")
    print(f"\ny1 = Relu(x):\n{y1}")
    print(f"y2 = Sigmoid(x):\n{y2}")
    print(f"y3 = Tanh(x):\n{y3}")
    print(f"\nThen branch (y1+y2+y3):\n{out_then}")
    print(f"Expected:\n{expected_then}")
    print(f"Match: {np.allclose(out_then, expected_then)}")
    print(f"\nElse branch (y1*y2*y3):\n{out_else}")
    print(f"Expected:\n{expected_else}")
    print(f"Match: {np.allclose(out_else, expected_else)}")

    assert np.allclose(out_then, expected_then), "Then branch mismatch!"
    assert np.allclose(out_else, expected_else), "Else branch mismatch!"
    print("\nAll tests passed!")


def main():
    model = build_model()
    onnx.save(model, 'outer_scope_multi_var.onnx')
    print("Saved outer_scope_multi_var.onnx")

    test_model(model)


if __name__ == '__main__':
    main()
