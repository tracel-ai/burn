#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    # Create a model that compares two Shape outputs
    # Input: tensor of shape [2, 3, 4]
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    
    # Shape node 1: Get shape of input
    shape1 = helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape1"]
    )
    
    # Constant shape for comparison
    const_shape = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_shape"],
        value=helper.make_tensor(
            name="const_value",
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[2, 3, 4]
        )
    )
    
    # Equal: Compare the two shapes
    equal = helper.make_node(
        "Equal",
        inputs=["shape1", "const_shape"],
        outputs=["output"]
    )
    
    # Output: boolean tensor
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [3])
    
    # Create the graph
    graph = helper.make_graph(
        [shape1, const_shape, equal],
        "equal_shape_test",
        [input_tensor],
        [output]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 16
    
    # Save the model
    onnx.save(model, "equal_shape.onnx")
    print("Model saved as equal_shape.onnx")

if __name__ == "__main__":
    main()