#!/usr/bin/env python3

import onnx
import torch
import torch.nn as nn
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    # Build ONNX graph that reshapes a Shape(3) to Shape(3) 
    # This tests the Shape -> Shape path in Reshape
    
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.INT64, [3]
    )
    
    # Create reshape target as [3] - reshape Shape(3) to Shape(3)
    reshape_target = numpy_helper.from_array(torch.tensor([3], dtype=torch.int64).numpy(), name="reshape_target")
    
    nodes = [
        # Get shape of input (will be Shape(3) containing [2, 3, 4])
        helper.make_node(
            "Shape",
            inputs=["input"],
            outputs=["input_shape"],
            name="shape1"
        ),
        # Reshape the Shape(3) to Shape(3) - essentially a no-op but tests the path
        helper.make_node(
            "Reshape",
            inputs=["input_shape", "reshape_target"],
            outputs=["output"],
            name="reshape1"
        ),
    ]
    
    graph = helper.make_graph(
        nodes,
        "reshape_shape_to_shape",
        [input_tensor],
        [output],
        initializer=[reshape_target]
    )
    
    onnx_model = helper.make_model(
        graph,
        producer_name="reshape_shape_to_shape_test",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    
    # Save the model
    onnx.save(onnx_model, "reshape_shape_to_shape.onnx")
    print("Model saved to reshape_shape_to_shape.onnx")
    
    # Test with ReferenceEvaluator
    try:
        test_input = np.random.randn(2, 3, 4).astype(np.float32)
        print(f"\nTest input shape: {test_input.shape}")
        
        session = ReferenceEvaluator(onnx_model, verbose=0)
        output, = session.run(None, {"input": test_input})
        
        print(f"Output: {output}")
        print(f"Output shape: {output.shape}")
        print(f"Expected: [2, 3, 4] (Shape(3) -> Shape(3) no-op)")
        
        # Verify the result
        expected = np.array([2, 3, 4], dtype=np.int64)
        assert np.array_equal(output, expected), f"Expected {expected}, got {output}"
        print("Test passed: Shape(3) to Shape(3) reshape worked correctly")
        
    except Exception as e:
        print(f"ReferenceEvaluator error: {e}")


if __name__ == "__main__":
    main()