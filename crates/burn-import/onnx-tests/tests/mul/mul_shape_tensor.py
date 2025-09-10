#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/mul/mul_shape_tensor.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16

def main():
    # Create a graph that tests Shape*Tensor and Tensor*Shape operations
    # Input tensors
    input_tensor = helper.make_tensor_value_info('input_tensor', TensorProto.FLOAT, [3, 4, 5])
    input_tensor_1d = helper.make_tensor_value_info('input_1d', TensorProto.INT64, [3])
    
    # Shape node - extract shape of input
    shape_node = helper.make_node(
        'Shape',
        inputs=['input_tensor'],
        outputs=['shape']
    )
    
    # Multiply shape with tensor (Shape * Tensor)
    mul_shape_tensor = helper.make_node(
        'Mul',
        inputs=['shape', 'input_1d'],
        outputs=['shape_times_tensor']
    )
    
    # Multiply tensor with shape (Tensor * Shape) - reversed order
    mul_tensor_shape = helper.make_node(
        'Mul',
        inputs=['input_1d', 'shape'],
        outputs=['tensor_times_shape']
    )
    
    # Outputs
    output1 = helper.make_tensor_value_info('shape_times_tensor', TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info('tensor_times_shape', TensorProto.INT64, [3])
    
    # Create the graph
    graph_def = helper.make_graph(
        [shape_node, mul_shape_tensor, mul_tensor_shape],
        'shape_tensor_mul_test',
        [input_tensor, input_tensor_1d],
        [output1, output2],
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name='onnx-tests',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Save the model
    onnx_name = "mul_shape_tensor.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))
    
    # Test the model with sample data
    test_input = np.random.randn(3, 4, 5).astype(np.float32)
    test_1d = np.array([2, 3, 2], dtype=np.int64)
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test 1d tensor: {test_1d}")
    
    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input, "input_1d": test_1d})
    
    shape_times_tensor, tensor_times_shape = outputs
    
    print(f"\nTest output shape_times_tensor: {repr(shape_times_tensor)}")
    print(f"Test output tensor_times_shape: {repr(tensor_times_shape)}")
    
    # Verify results are the same (multiplication is commutative)
    assert np.array_equal(shape_times_tensor, tensor_times_shape), "Multiplication should be commutative"

if __name__ == '__main__':
    main()