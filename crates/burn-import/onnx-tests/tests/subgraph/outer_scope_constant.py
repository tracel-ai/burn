#!/usr/bin/env python3
"""
Generate ONNX model that tests outer-scope CONSTANT/INITIALIZER references in subgraphs.

This tests the critical scenario where a subgraph (If branch) references constants
or initializers defined in the PARENT graph, not local to the subgraph.

This is different from outer_scope_ref.py which tests outer-scope computed values.
Here we specifically test that:
1. Constants defined in parent graph can be accessed inside If branches
2. Initializers (weights) defined in parent graph can be accessed inside If branches

This pattern is common in real models like Silero VAD where Conv layers inside
If branches use weights from the parent graph.

Pattern:
    # Parent graph has:
    weight = constant [2, 3]  # MatMul weight in parent
    bias = constant [2]       # Add bias in parent

    z = If(condition) {
        then: MatMul(x, weight) + bias  # Uses parent's weight/bias
        else: x * 2                      # Simple fallback
    }
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
    Build If model with outer-scope constant/initializer references.

    The key difference from outer_scope_ref.py is that we reference
    INITIALIZERS (weight, bias) from the parent graph, not computed values.
    """
    # Parent graph initializers - these are the weights we want to access from subgraph
    # Weight for MatMul: input [1, 3] @ weight [3, 2] = output [1, 2]
    weight_data = np.array([[1.0, 0.5],
                            [2.0, 1.0],
                            [0.5, 2.0]], dtype=np.float32)  # [3, 2]
    bias_data = np.array([0.1, 0.2], dtype=np.float32)  # [2]

    weight_init = numpy_helper.from_array(weight_data, name='weight')
    bias_init = numpy_helper.from_array(bias_data, name='bias')

    # Then branch: MatMul + Add using parent's weight and bias
    # Note: 'weight' and 'bias' are NOT declared as inputs - they come from outer scope
    then_matmul = helper.make_node('MatMul',
                                    inputs=['x', 'weight'],
                                    outputs=['matmul_out'])
    then_add = helper.make_node('Add',
                                 inputs=['matmul_out', 'bias'],
                                 outputs=['then_out'])
    then_graph = helper.make_graph(
        nodes=[then_matmul, then_add],
        name='then_branch',
        inputs=[],  # No explicit inputs - weight/bias come from outer scope
        outputs=[helper.make_tensor_value_info('then_out', TensorProto.FLOAT, [1, 2])],
    )

    # Else branch: Simple multiply by 2
    # Need to match output shape [1, 2], so use a different x_else input
    else_mul = helper.make_node('Mul', inputs=['x_slice', 'scale'], outputs=['else_out'])
    else_graph = helper.make_graph(
        nodes=[else_mul],
        name='else_branch',
        inputs=[],
        outputs=[helper.make_tensor_value_info('else_out', TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='scale')]
    )

    # Main graph
    # Slice x to get first 2 elements for else branch (to match output shape)
    slice_node = helper.make_node('Slice',
                                   inputs=['x', 'starts', 'ends', 'axes'],
                                   outputs=['x_slice'])
    if_node = helper.make_node('If', inputs=['condition'], outputs=['output'],
                                then_branch=then_graph, else_branch=else_graph)

    main_graph = helper.make_graph(
        nodes=[slice_node, if_node],
        name='outer_scope_constant',
        inputs=[
            helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info('condition', TensorProto.BOOL, []),
        ],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
        initializer=[
            weight_init,  # Parent graph constant - accessed from subgraph
            bias_init,    # Parent graph constant - accessed from subgraph
            numpy_helper.from_array(np.array([0], dtype=np.int64), name='starts'),
            numpy_helper.from_array(np.array([2], dtype=np.int64), name='ends'),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name='axes'),
        ]
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

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # [1, 3]

    # Weight: [3, 2]
    weight = np.array([[1.0, 0.5],
                       [2.0, 1.0],
                       [0.5, 2.0]], dtype=np.float32)
    bias = np.array([0.1, 0.2], dtype=np.float32)

    sess = ort.InferenceSession(model.SerializeToString())

    out_then = sess.run(None, {'x': x, 'condition': np.array(True, dtype=bool)})[0]
    out_else = sess.run(None, {'x': x, 'condition': np.array(False, dtype=bool)})[0]

    # Then branch: x @ weight + bias
    # [1, 3] @ [3, 2] = [1, 2]
    # [[1, 2, 3]] @ [[1, 0.5], [2, 1], [0.5, 2]] = [[1*1 + 2*2 + 3*0.5, 1*0.5 + 2*1 + 3*2]]
    #                                            = [[1 + 4 + 1.5, 0.5 + 2 + 6]] = [[6.5, 8.5]]
    # + bias [0.1, 0.2] = [[6.6, 8.7]]
    expected_then = x @ weight + bias

    # Else branch: x[:, :2] * 2 = [[1, 2]] * 2 = [[2, 4]]
    expected_else = x[:, :2] * 2

    print("=== Outer Scope Constant Test ===")
    print(f"x: {x}")
    print(f"\nThen branch (MatMul + Add with outer-scope weight/bias):")
    print(f"Output: {out_then}")
    print(f"Expected: {expected_then}")
    print(f"Match: {np.allclose(out_then, expected_then)}")
    print(f"\nElse branch (x[:,:2]*2):")
    print(f"Output: {out_else}")
    print(f"Expected: {expected_else}")
    print(f"Match: {np.allclose(out_else, expected_else)}")

    # Print values for Rust test
    print("\n=== Values for Rust test ===")
    print(f"Input x: {x.flatten().tolist()}")
    print(f"Then output: {out_then.flatten().tolist()}")
    print(f"Else output: {out_else.flatten().tolist()}")

    assert np.allclose(out_then, expected_then, atol=1e-5), f"Then branch mismatch!\nGot: {out_then}\nExpected: {expected_then}"
    assert np.allclose(out_else, expected_else), "Else branch mismatch!"
    print("\nAll tests passed!")


def main():
    model = build_model()
    onnx.save(model, 'outer_scope_constant.onnx')
    print("Saved outer_scope_constant.onnx")

    test_model(model)


if __name__ == '__main__':
    main()
