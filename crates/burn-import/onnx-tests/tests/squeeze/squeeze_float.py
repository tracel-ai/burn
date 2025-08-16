#!/usr/bin/env python3

# used to generate model: squeeze_float.onnx
# Test squeeze operation for float tensors to scalars

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    # Create a model that squeezes a 1D float tensor to a scalar
    # Input: 1D float tensor with one element
    input_tensor = helper.make_tensor_value_info(
        'input', 
        TensorProto.FLOAT, 
        [1]  # 1D tensor with size 1
    )
    
    # Output: scalar (0-dimensional tensor)
    output_scalar = helper.make_tensor_value_info(
        'output', 
        TensorProto.FLOAT, 
        []  # scalar has no dimensions
    )
    
    # Create axes constant for squeeze (squeeze dimension 0)
    axes_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=helper.make_tensor(
            name='axes_value',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    )
    
    # Squeeze operation: tensor[1] -> scalar
    squeeze_node = helper.make_node(
        'Squeeze',
        inputs=['input', 'axes'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        [axes_const, squeeze_node],
        'squeeze_float',
        [input_tensor],
        [output_scalar]
    )
    
    # Create the model
    model = helper.make_model(
        graph, 
        producer_name="squeeze_float",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    model.ir_version = 8  # Use IR version 8 for compatibility
    
    # Save the model
    onnx.save(model, "squeeze_float.onnx")
    print("Finished exporting model to squeeze_float.onnx")
    
    # Verify the model
    onnx.checker.check_model(model)
    
    # Test with ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        session = ReferenceEvaluator(model, verbose=0)
        
        # Test input: 1D float array with one element
        test_input = np.array([3.14159], dtype=np.float32)
        print(f"\nTest input data: {test_input}")
        print(f"Test input shape: {test_input.shape}")
        
        # Run inference
        output, = session.run(None, {"input": test_input})
        
        print(f"Test output data: {output}")
        print(f"Test output shape: {output.shape}")
        print(f"Test output type: {output.dtype}")
        
        # Verify it's a scalar
        assert output.shape == (), "Output should be a scalar (0-dimensional)"
        assert np.isclose(output, test_input[0]), "Value should be preserved"
        print("Test passed: float tensor successfully squeezed to scalar")
        
    except ImportError:
        print("onnx.reference not available, skipping ReferenceEvaluator test")
        # Fallback simulation
        test_input = np.array([3.14159], dtype=np.float32)
        print(f"\nTest input data: {test_input}")
        print(f"Test input shape: {test_input.shape}")
        
        # Simulate squeeze operation
        output = test_input.squeeze(axis=0)
        print(f"Simulated output data: {output}")
        print(f"Simulated output shape: {output.shape}")
        print(f"Simulated output type: {output.dtype}")


if __name__ == "__main__":
    main()