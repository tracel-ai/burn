#!/usr/bin/env python3

# used to generate model: concat_shape.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create Shape nodes to extract shapes from input tensors
    shape1_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="/Shape1"
    )
    
    shape2_node = onnx.helper.make_node(
        "Shape",
        inputs=["input2"],
        outputs=["shape2"],
        name="/Shape2"
    )
    
    shape3_node = onnx.helper.make_node(
        "Shape",
        inputs=["input3"],
        outputs=["shape3"],
        name="/Shape3"
    )
    
    # Create a Concat node that concatenates the shapes
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["shape1", "shape2", "shape3"],
        outputs=["concatenated_shape"],
        axis=0,  # Required attribute
        name="/Concat"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[shape1_node, shape2_node, shape3_node, concat_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3]
                ),
            ),
            onnx.helper.make_value_info(
                name="input2",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[4, 5, 6]
                ),
            ),
            onnx.helper.make_value_info(
                name="input3",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[7]
                ),
            )
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="concatenated_shape",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[6]  # 2 + 3 + 1 = 6 dimensions total
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
    file_name = "concat_shape.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input1 = np.ones((2, 3), dtype=np.float32)
        test_input2 = np.ones((4, 5, 6), dtype=np.float32)
        test_input3 = np.ones((7,), dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {
            "input1": test_input1, 
            "input2": test_input2,
            "input3": test_input3
        })
        
        print(f"Test input1 shape: {test_input1.shape}")
        print(f"Test input2 shape: {test_input2.shape}")
        print(f"Test input3 shape: {test_input3.shape}")
        print(f"Concatenated shape output: {result[0]}")
        print(f"Expected: [2, 3, 4, 5, 6, 7]")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()