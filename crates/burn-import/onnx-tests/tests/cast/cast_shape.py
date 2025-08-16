#!/usr/bin/env python3

# used to generate model: cast_shape.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape_original"],
        name="/Shape"
    )
    
    # Cast the shape output to int32 (this should be a no-op in Burn, staying as [i64; N])
    cast_to_int32_node = onnx.helper.make_node(
        "Cast",
        inputs=["shape_original"],
        outputs=["shape_casted"],
        to=onnx.TensorProto.INT32,
        name="/Cast_to_int32"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[shape_node, cast_to_int32_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3, 4]
                ),
            )
        ],
        outputs=[
            # Output both the original and casted shapes
            onnx.helper.make_value_info(
                name="shape_original",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[3]
                ),
            ),
            onnx.helper.make_value_info(
                name="shape_casted",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT32, shape=[3]
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
    file_name = "cast_shape.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.ones((2, 3, 4), dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        results = sess.run(None, {"input": test_input})
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Shape (original i64): {results[0]} dtype={results[0].dtype}")
        print(f"Shape (casted to i32): {results[1]} dtype={results[1].dtype}")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()