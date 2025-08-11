#!/usr/bin/env python3

import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper


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
    
    onnx_model = helper.make_model(graph)
    onnx_model.opset_import[0].version = 16
    
    # Save the model
    onnx.save(onnx_model, "reshape_shape_to_shape.onnx")
    print("Model saved to reshape_shape_to_shape.onnx")


if __name__ == "__main__":
    main()