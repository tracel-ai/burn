#!/usr/bin/env python3
"""
Generate ONNX model that tests outer-scope variable references in subgraphs.

This tests the DeferredGraph lazy building pattern where subgraphs reference
values computed in parent graphs.

Pattern:
    y = Relu(x)           # Parent computes y
    z = If(condition) {
        then: Add(y, 10)  # References y from parent
        else: Mul(y, 2)   # References y from parent
    }

The subgraph references 'y' WITHOUT declaring it as an explicit input.
This requires DeferredGraph to resolve the type from outer scope.
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("Warning: onnxruntime not available, skipping runtime validation")


def build_model():
    """
    Build If model with outer-scope reference.

    Structure:
        y = Relu(x)           # Parent computes y
        z = If(condition) {
            then: Add(y, 10)  # References y from parent
            else: Mul(y, 2)   # References y from parent
        }
    """
    # Then branch: y + 10 (y comes from outer scope)
    then_add = helper.make_node('Add', inputs=['y', 'bias'], outputs=['then_out'])
    then_graph = helper.make_graph(
        nodes=[then_add],
        name='then_branch',
        inputs=[],  # No explicit inputs - y comes from outer scope
        outputs=[helper.make_tensor_value_info('then_out', TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(np.array([10.0], dtype=np.float32), name='bias')]
    )

    # Else branch: y * 2 (y comes from outer scope)
    else_mul = helper.make_node('Mul', inputs=['y', 'scale'], outputs=['else_out'])
    else_graph = helper.make_graph(
        nodes=[else_mul],
        name='else_branch',
        inputs=[],  # No explicit inputs - y comes from outer scope
        outputs=[helper.make_tensor_value_info('else_out', TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='scale')]
    )

    # Main graph
    relu_node = helper.make_node('Relu', inputs=['x'], outputs=['y'])
    if_node = helper.make_node('If', inputs=['condition'], outputs=['output'],
                                then_branch=then_graph, else_branch=else_graph)

    main_graph = helper.make_graph(
        nodes=[relu_node, if_node],
        name='outer_scope_ref',
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


def test_model(model):
    """Test the model with ONNX Runtime."""
    if not HAS_ORT:
        print("Skipping runtime test (onnxruntime not available)")
        return

    x = np.array([[-1.5, 2.0, -0.5], [3.0, -2.0, 1.0]], dtype=np.float32)

    sess = ort.InferenceSession(model.SerializeToString())

    # y = Relu(x) = [[0, 2, 0], [3, 0, 1]]
    y = np.maximum(x, 0)

    out_then = sess.run(None, {'x': x, 'condition': np.array(True, dtype=bool)})[0]
    out_else = sess.run(None, {'x': x, 'condition': np.array(False, dtype=bool)})[0]

    expected_then = y + 10
    expected_else = y * 2

    print("=== Outer Scope Reference Test ===")
    print(f"x:\n{x}")
    print(f"y = Relu(x):\n{y}")
    print(f"\nThen branch (y+10):\n{out_then}")
    print(f"Expected:\n{expected_then}")
    print(f"Match: {np.allclose(out_then, expected_then)}")
    print(f"\nElse branch (y*2):\n{out_else}")
    print(f"Expected:\n{expected_else}")
    print(f"Match: {np.allclose(out_else, expected_else)}")

    assert np.allclose(out_then, expected_then), "Then branch mismatch!"
    assert np.allclose(out_else, expected_else), "Else branch mismatch!"
    print("\nAll tests passed!")


def main():
    model = build_model()
    onnx.save(model, 'outer_scope_ref.onnx')
    print("Saved outer_scope_ref.onnx")

    test_model(model)


if __name__ == '__main__':
    main()
