#!/usr/bin/env python3

# Test that ensures constant tensors are properly converted to Shape type
# when used in binary operations with Shape inputs, especially during propagation

import onnx
import onnx.helper
import numpy as np

def build_model():
    # Create the graph nodes
    
    # Shape node to extract shape from input tensor
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=["input1"],
        outputs=["shape1"],
        name="shape_node"
    )
    
    # Constant to add to the shape (1D tensor)
    add_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["add_constant"],
        value=onnx.helper.make_tensor(
            name="const_add_val",
            data_type=onnx.TensorProto.INT64,
            dims=[3],
            vals=[10, 20, 30]
        ),
        name="add_constant_node"
    )
    
    # Add operation - shape1 is Shape, so add_constant should be converted to Shape
    add_node = onnx.helper.make_node(
        "Add",
        inputs=["shape1", "add_constant"],
        outputs=["add_result"],
        name="add_node"
    )
    
    # Constant for division (this tests the propagation case)
    # After add_result becomes Shape, this constant should also be converted
    div_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["div_constant"],
        value=onnx.helper.make_tensor(
            name="const_div_val",
            data_type=onnx.TensorProto.INT64,
            dims=[3],
            vals=[2, 2, 2]
        ),
        name="div_constant_node"
    )
    
    # Div operation - add_result is Shape, so div_constant should be converted to Shape
    div_node = onnx.helper.make_node(
        "Div",
        inputs=["add_result", "div_constant"],
        outputs=["div_result"],
        name="div_node"
    )
    
    # Constant for subtraction
    sub_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sub_constant"],
        value=onnx.helper.make_tensor(
            name="const_sub_val",
            data_type=onnx.TensorProto.INT64,
            dims=[3],
            vals=[3, 4, 5]
        ),
        name="sub_constant_node"
    )
    
    # Sub operation
    sub_node = onnx.helper.make_node(
        "Sub",
        inputs=["div_result", "sub_constant"],
        outputs=["sub_result"],
        name="sub_node"
    )
    
    # Constant for multiplication
    mul_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mul_constant"],
        value=onnx.helper.make_tensor(
            name="const_mul_val",
            data_type=onnx.TensorProto.INT64,
            dims=[3],
            vals=[4, 5, 6]
        ),
        name="mul_constant_node"
    )
    
    # Mul operation
    mul_node = onnx.helper.make_node(
        "Mul",
        inputs=["sub_result", "mul_constant"],
        outputs=["final_result"],
        name="mul_node"
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        name="shape_binary_ops_test",
        nodes=[
            shape_node,
            add_const, add_node,
            div_const, div_node,
            sub_const, sub_node,
            mul_const, mul_node
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 8, 3]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="final_result",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[3]
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
    file_name = "shape_binary_ops_with_constant.onnx"
    onnx.save(onnx_model, file_name)
    onnx.checker.check_model(file_name)
    
    print(f"Finished exporting model to {file_name}")
    
    # Test with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Create test data
        test_input = np.ones((2, 8, 3), dtype=np.float32)
        
        # Run inference
        sess = ReferenceEvaluator(onnx_model)
        result = sess.run(None, {"input1": test_input})
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Shape operations:")
        print(f"  shape = [2, 8, 3]")
        print(f"  + [10, 20, 30] = [12, 28, 33]")
        print(f"  / [2, 2, 2] = [6, 14, 16]")
        print(f"  - [3, 4, 5] = [3, 10, 11]")
        print(f"  * [4, 5, 6] = [12, 50, 66]")
        print(f"Final result: {result[0]}")
        
        expected = np.array([12, 50, 66], dtype=np.int64)
        assert np.array_equal(result[0], expected), f"Expected {expected}, got {result[0]}"
        print("Test passed!")
        
    except ImportError:
        print("onnx.reference not available, skipping inference test")


if __name__ == "__main__":
    main()