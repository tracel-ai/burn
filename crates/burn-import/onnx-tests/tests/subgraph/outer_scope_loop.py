#!/usr/bin/env python3
"""
Generate ONNX model that tests outer-scope variable references in Loop subgraphs.

Pattern:
    y = Relu(x)           # Parent computes y
    z = Loop(3) {
        body: accum = Add(accum, y)  # References y from parent (not a loop input!)
    }

The loop body references 'y' WITHOUT declaring it as an input.
Result: accum_final = 0 + y + y + y = 3*y
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
    Build Loop model with outer-scope reference.

    Structure:
        y = Relu(x)                    # Parent computes y
        z = Loop(3) {
            body: accum = Add(accum, y)  # References y from parent (not a loop input!)
        }
    """
    # Loop body: accum + y (y comes from outer scope, not passed as loop input)
    body_add = helper.make_node('Add', inputs=['accum', 'y'], outputs=['accum_out'])
    body_cond = helper.make_node('Identity', inputs=['cond_in'], outputs=['cond_out'])

    body_graph = helper.make_graph(
        nodes=[body_add, body_cond],
        name='loop_body',
        inputs=[
            helper.make_tensor_value_info('iter', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_in', TensorProto.BOOL, []),
            helper.make_tensor_value_info('accum', TensorProto.FLOAT, [2, 3]),
            # Note: 'y' is NOT declared as input - it's an outer-scope reference
        ],
        outputs=[
            helper.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
            helper.make_tensor_value_info('accum_out', TensorProto.FLOAT, [2, 3]),
        ],
    )

    # Main graph
    relu_node = helper.make_node('Relu', inputs=['x'], outputs=['y'])

    # Loop: 3 iterations, initial accum = zeros
    loop_node = helper.make_node(
        'Loop',
        inputs=['max_iter', 'cond_init', 'accum_init'],
        outputs=['output'],
        body=body_graph
    )

    main_graph = helper.make_graph(
        nodes=[relu_node, loop_node],
        name='outer_scope_loop',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('max_iter', TensorProto.INT64, []),
            helper.make_tensor_value_info('cond_init', TensorProto.BOOL, []),
            helper.make_tensor_value_info('accum_init', TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])],
    )

    model = helper.make_model(main_graph, producer_name='burn-import-test',
                               opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def test_model(model):
    """Test the model with ONNX Runtime."""
    if not HAS_ORT:
        print("Skipping runtime test (onnxruntime not available)")
        return

    x = np.array([[-1.5, 2.0, -0.5], [3.0, -2.0, 1.0]], dtype=np.float32)
    max_iter = np.array(3, dtype=np.int64)
    cond_init = np.array(True, dtype=bool)
    accum_init = np.zeros([2, 3], dtype=np.float32)

    sess = ort.InferenceSession(model.SerializeToString())

    # y = Relu(x) = [[0, 2, 0], [3, 0, 1]]
    y = np.maximum(x, 0)

    out = sess.run(None, {
        'x': x,
        'max_iter': max_iter,
        'cond_init': cond_init,
        'accum_init': accum_init
    })[0]

    # Loop runs 3 times: accum = 0 + y + y + y = 3*y
    expected = 3 * y

    print("=== Loop Outer Scope Reference Test ===")
    print(f"x:\n{x}")
    print(f"y = Relu(x):\n{y}")
    print(f"Output (0 + y + y + y):\n{out}")
    print(f"Expected (3*y):\n{expected}")
    print(f"Match: {np.allclose(out, expected)}")

    assert np.allclose(out, expected), "Loop output mismatch!"
    print("\nTest passed!")


def main():
    model = build_model()
    onnx.save(model, 'outer_scope_loop.onnx')
    print("Saved outer_scope_loop.onnx")

    test_model(model)


if __name__ == '__main__':
    main()
