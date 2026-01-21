#!/usr/bin/env python3

# used to generate model: cast_shape_to_float.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input"],
        outputs=["shape_output"],
        name="/Shape"
    )
    
    # Cast the shape output to float32 (this should convert Shape to 1D tensor)
    cast_to_float_node = onnx.helper.make_node(
        "Cast",
        inputs=["shape_output"],
        outputs=["cast_output"],
        to=onnx.TensorProto.FLOAT,
        name="/Cast_to_float"
    )
    
    # Multiply by a scalar input to verify it's a tensor (not a Shape)
    mul_node = onnx.helper.make_node(
        "Mul",
        inputs=["cast_output", "multiplier"],
        outputs=["output"],
        name="/Mul"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[shape_node, cast_to_float_node, mul_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3, 4]
                ),
            ),
            onnx.helper.make_value_info(
                name="multiplier",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[1]
                ),
            )
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[3]
                ),
            )
        ],
        initializer=[]
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
    file_name = "cast_shape_to_float.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.ones((2, 3, 4), dtype=np.float32)
        multiplier = np.array([2.0], dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        results = sess.run(None, {"input": test_input, "multiplier": multiplier})
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Output (shape * 2): {results[0]} dtype={results[0].dtype}")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()