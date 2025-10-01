#!/usr/bin/env python3

# used to generate model: reshape_with_1d_tensor.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a reshape node that takes shape as a tensor input
    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="/Reshape"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[reshape_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[12]
                ),
            ),
            onnx.helper.make_value_info(
                name="shape",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[2]
                ),
            )
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[3, 4]
                ),
            )
        ]
    )
    
    # Create the model
    model = onnx.helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)]
    )
    
    return model


def main():
    onnx_model = build_model()
    file_name = "reshape_with_1d_tensor.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.arange(12, dtype=np.float32)
        test_shape = np.array([3, 4], dtype=np.int64)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input": test_input, "shape": test_shape})
        
        print(f"Test input data: {test_input}")
        print(f"Test input data shape: {test_input.shape}")
        print(f"Test shape tensor: {test_shape}")
        print(f"Test output data shape: {result[0].shape}")
        print(f"Test output:\n{result[0]}")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()