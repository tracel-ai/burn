#!/usr/bin/env python3

import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper


def main():
    # Build ONNX graph that reshapes a Shape from Shape(4) to Shape(2)
    # This tests partial reshaping of Shape arrays
    
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4, 5]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.INT64, [2]
    )
    
    # Create reshape target to reshape from 4 elements to 2 elements [2, 2]
    reshape_target = numpy_helper.from_array(torch.tensor([2], dtype=torch.int64).numpy(), name="reshape_target")
    
    nodes = [
        # Get shape of input (will be Shape(4) containing [2, 3, 4, 5])
        helper.make_node(
            "Shape",
            inputs=["input"],
            outputs=["input_shape"],
            name="shape1"
        ),
        # Slice to get first 4 elements (in this case all of them)
        helper.make_node(
            "Slice",
            inputs=["input_shape", "slice_start", "slice_end"],
            outputs=["sliced_shape"],
            name="slice1"
        ),
        # Reshape the Shape(4) to Shape(2) - [2,3,4,5] becomes [2,3] (takes first 2 elements)
        helper.make_node(
            "Reshape",
            inputs=["sliced_shape", "reshape_target"],
            outputs=["output"],
            name="reshape1"
        ),
    ]
    
    # Add slice start/end constants
    start = numpy_helper.from_array(torch.tensor([0], dtype=torch.int64).numpy(), name="slice_start")
    end = numpy_helper.from_array(torch.tensor([2], dtype=torch.int64).numpy(), name="slice_end")
    
    graph = helper.make_graph(
        nodes,
        "reshape_shape_partial",
        [input_tensor],
        [output],
        initializer=[start, end, reshape_target]
    )
    
    onnx_model = helper.make_model(
        graph,
        producer_name="reshape_shape_partial_test",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    
    # Save the model
    onnx.save(onnx_model, "reshape_shape_partial.onnx")
    print("Model saved to reshape_shape_partial.onnx")


if __name__ == "__main__":
    main()