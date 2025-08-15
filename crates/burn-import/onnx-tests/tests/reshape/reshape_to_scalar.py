#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Create a simple model that reshapes a 1x1 tensor to a scalar
    # Input: 1x1 tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    
    # Output: scalar (rank-0 tensor)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
    
    # Create shape tensor for reshape (empty shape = scalar)
    shape_tensor = helper.make_tensor(
        name="shape",
        data_type=TensorProto.INT64,
        dims=[0],  # Shape tensor is 0-dimensional
        vals=[]    # Empty values for scalar shape
    )
    
    # Create Reshape node
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="reshape_to_scalar"
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        [reshape_node],
        "reshape_to_scalar_model",
        [input_tensor],
        [output_tensor],
        [shape_tensor]  # Shape is an initializer
    )
    
    # Create the model
    model_def = helper.make_model(
        graph_def, 
        producer_name="reshape_to_scalar",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model_def, "reshape_to_scalar.onnx")
    print("Model exported successfully to reshape_to_scalar.onnx")
    print("Model structure: Reshape([1, 1] -> scalar)")
    
    # Verify with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        test_input = np.array([[1.5]], dtype=np.float32)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input value: {test_input}")
        
        # Run inference with ONNX model
        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"input": test_input})
        
        print(f"ONNX model output shape: {result[0].shape}")
        print(f"ONNX model output value: {result[0]}")
        print(f"ONNX model output dtype: {result[0].dtype}")
        
    except ImportError:
        print("onnx.reference not available, skipping ONNX model verification")
        # Fallback to numpy verification
        reshaped = test_input.reshape(())  # Reshape to scalar
        print(f"NumPy reshaped shape: {reshaped.shape}")
        print(f"NumPy reshaped value: {reshaped}")


if __name__ == "__main__":
    main()