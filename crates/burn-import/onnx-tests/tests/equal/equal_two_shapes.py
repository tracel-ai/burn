#!/usr/bin/env python3

import onnx
import numpy as np
from onnx import helper, TensorProto

def main():
    # Create a model that compares shapes from two different tensors
    # Input tensors with same shape
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [2, 3, 4])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [2, 3, 4])
    
    # Shape node 1: Get shape of input1
    shape1 = helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"]
    )
    
    # Shape node 2: Get shape of input2
    shape2 = helper.make_node(
        "Shape",
        inputs=["input2"],
        outputs=["shape2"]
    )
    
    # Equal: Compare the two shapes
    equal = helper.make_node(
        "Equal",
        inputs=["shape1", "shape2"],
        outputs=["output"]
    )
    
    # Output: boolean tensor
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [3])
    
    # Create the graph
    graph = helper.make_graph(
        [shape1, shape2, equal],
        "equal_two_shapes_test",
        [input1, input2],
        [output]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 16
    
    # Save the model
    onnx.save(model, "equal_two_shapes.onnx")
    print("Model saved as equal_two_shapes.onnx")

if __name__ == "__main__":
    main()