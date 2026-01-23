#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/gelu/gelu.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

def main():
    # Create a simple ONNX graph with a native GELU op
    # Using opset 20 where GELU is a native op

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 1, 4])

    # Create GELU node with default approximate="none" (exact GELU)
    gelu_node = helper.make_node(
        'Gelu',
        inputs=['X'],
        outputs=['Y'],
        name='gelu1'
    )

    graph_def = helper.make_graph(
        [gelu_node],
        'gelu_test',
        [X],
        [Y]
    )

    model_def = helper.make_model(graph_def, producer_name='burn-onnx-test')
    model_def.opset_import[0].version = 20  # GELU requires opset 20

    # Validate the model
    onnx.checker.check_model(model_def)

    # Save the model
    onnx_name = "gelu.onnx"
    onnx.save(model_def, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Use reference evaluator to compute expected output
    test_input = np.array([[[[1.0, 4.0, 9.0, 25.0]]]], dtype=np.float32)
    print(f"Test input data: {test_input}")

    ref = ReferenceEvaluator(model_def)
    output = ref.run(None, {'X': test_input})[0]
    print(f"Test output data: {output}")
    print(f"Output values for Rust test: {output.flatten().tolist()}")

if __name__ == '__main__':
    main()
