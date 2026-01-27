#!/usr/bin/env python3

# used to generate model: concat_shape_with_constant.onnx

import onnx
import onnx.helper
import numpy as np


def build_model():
    # Create a constant node with a rank-1 tensor
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_shape"],
        value=onnx.helper.make_tensor(
            name="const_value",
            data_type=onnx.TensorProto.INT64,
            dims=[2],
            vals=[10, 20]
        ),
        name="/Constant"
    )
    
    # Create Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="/Shape"
    )
    
    # Create a Concat node that concatenates the constant and the shape
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["shape1", "const_shape"],
        outputs=["concatenated_shape"],
        axis=0,  # Required attribute
        name="/Concat"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="main_graph",
        nodes=[const_node, shape_node, concat_node],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[3, 4, 5]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="concatenated_shape",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[5]  # 3 + 2 = 5 dimensions total
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
    file_name = "concat_shape_with_constant.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.ones((3, 4, 5), dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input1": test_input})
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Concatenated shape output: {result[0]}")
        print(f"Expected: [3, 4, 5, 10, 20]")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()