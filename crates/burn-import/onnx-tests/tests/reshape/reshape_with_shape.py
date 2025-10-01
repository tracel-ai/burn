#!/usr/bin/env python3

# used to generate model: reshape_with_shape.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["shape_source"],
        outputs=["extracted_shape"],
        name="/Shape"
    )
    
    # Create a Reshape node that uses the extracted shape
    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["input", "extracted_shape"],
        outputs=["output"],
        name="/Reshape"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[shape_node, reshape_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[12]
                ),
            ),
            onnx.helper.make_value_info(
                name="shape_source",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[3, 4]
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
    file_name = "reshape_with_shape.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.arange(12, dtype=np.float32)
        test_shape_source = np.zeros((3, 4), dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input": test_input, "shape_source": test_shape_source})
        
        print(f"Test input data: {test_input}")
        print(f"Test input data shape: {test_input.shape}")
        print(f"Test shape source shape: {test_shape_source.shape}")
        print(f"Test output data shape: {result[0].shape}")
        print(f"Test output:\n{result[0]}")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()