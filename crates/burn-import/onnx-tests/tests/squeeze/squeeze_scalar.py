#!/usr/bin/env python3

# This script generates ONNX model for testing squeeze of a scalar (no-op)
# using ONNX tools directly without PyTorch

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def main():
    # Create a constant scalar value
    scalar_value = np.array(1.5, dtype=np.float32)
    scalar_tensor = numpy_helper.from_array(scalar_value, name="scalar_const")
    
    # Create nodes
    nodes = [
        # Constant node that outputs a scalar
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scalar"],
            value=scalar_tensor
        ),
        # Squeeze node on the scalar (should be no-op)
        # In opset 16, axes is provided as input, not attribute
        helper.make_node(
            "Squeeze",
            inputs=["scalar"],  # No axes input means squeeze all dims of size 1
            outputs=["output"],
        ),
    ]
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        "main_graph",
        inputs=[],  # No inputs, using constant
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [])  # Scalar output
        ],
    )
    
    # Create the model with opset 16
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, "squeeze_scalar.onnx")
    print("Generated squeeze_scalar.onnx")


if __name__ == "__main__":
    main()