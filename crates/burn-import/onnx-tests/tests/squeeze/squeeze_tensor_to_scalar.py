#!/usr/bin/env python3

import numpy as np
import onnx
from onnx import TensorProto, helper

def main():
    # Create a multi-dimensional tensor input with shape that will have 1 element
    # e.g., shape [1, 1, 1] which has exactly 1 element
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1])
    
    # Squeeze all dimensions to get a scalar output
    # When all dimensions are squeezed from a [1,1,1] tensor, we get a scalar
    squeeze = helper.make_node(
        "Squeeze",
        inputs=["input"],
        outputs=["output"],
        axes=[0, 1, 2]  # Squeeze all three dimensions
    )
    
    # Output is a scalar (rank 0)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
    
    # Create the graph
    graph_def = helper.make_graph(
        [squeeze],
        "test_squeeze_to_scalar",
        [input_tensor],
        [output_tensor],
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name="squeeze_to_scalar_test")
    model_def.opset_import[0].version = 16
    
    # Save the model
    onnx.save(model_def, "squeeze_tensor_to_scalar.onnx")
    print("Model saved as squeeze_tensor_to_scalar.onnx")

if __name__ == "__main__":
    main()