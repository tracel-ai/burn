#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/resize/resize_with_shape.onnx

import onnx
from onnx import helper, TensorProto
import numpy as np

def main() -> None:
    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 3, 4, 4])
    
    # Create Shape node to get shape of input
    shape_node = helper.make_node(
        "Shape",
        name="shape_node",
        inputs=["input_tensor"],
        outputs=["input_shape"],
    )
    
    # Create constants for slicing the shape
    # We want to extract the first 2 dimensions (batch and channel)
    starts = helper.make_tensor(
        name="starts",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )
    
    ends = helper.make_tensor(
        name="ends",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[2],
    )
    
    axes = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0],
    )
    
    # Slice to get the first 2 dimensions [1, 3]
    slice_node = helper.make_node(
        "Slice",
        name="slice_node",
        inputs=["input_shape", "starts", "ends", "axes"],
        outputs=["batch_channel_dims"],
    )
    
    # Create constant for new spatial dimensions [8, 8]
    new_spatial_dims = helper.make_tensor(
        name="new_spatial_dims",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[8, 8],
    )
    
    # Concat to create new shape [1, 3, 8, 8]
    concat_node = helper.make_node(
        "Concat",
        name="concat_node",
        inputs=["batch_channel_dims", "new_spatial_dims"],
        outputs=["new_shape"],
        axis=0,
    )
    
    # Resize using the computed shape
    resize_node = helper.make_node(
        "Resize",
        name="resize_node",
        inputs=["input_tensor", "", "", "new_shape"],
        outputs=["output"],
        mode="linear",
    )
    
    graph_def = helper.make_graph(
        nodes=[shape_node, slice_node, concat_node, resize_node],
        name="ResizeWithShapeGraph",
        inputs=[input_tensor],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
        ],
        initializer=[starts, ends, axes, new_spatial_dims],
    )
    
    model_def = helper.make_model(
        graph_def, 
        producer_name="resize_with_shape",
        opset_imports=[helper.make_operatorsetid("", 16)]
    )
    
    onnx.save(model_def, "resize_with_shape.onnx")


    # Verify with onnx.reference.ReferenceEvaluator
    try:
        from onnx.reference import ReferenceEvaluator
        
        # Test input matching the Rust test
        test_input = np.array([[
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            [
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
                [24.0, 25.0, 26.0, 27.0],
                [28.0, 29.0, 30.0, 31.0],
            ],
            [
                [32.0, 33.0, 34.0, 35.0],
                [36.0, 37.0, 38.0, 39.0],
                [40.0, 41.0, 42.0, 43.0],
                [44.0, 45.0, 46.0, 47.0],
            ],
        ]], dtype=np.float32)
        
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input value: {test_input}")
        
        # Run inference with ONNX model
        sess = ReferenceEvaluator(model_def)
        result = sess.run(None, {"input_tensor": test_input})
        
        print(f"ONNX model output shape: {result[0].shape}")
        print(f"ONNX model output value: {result[0]}")
        print(f"ONNX model output dtype: {result[0].dtype}")
        
    except ImportError:
        print("onnx.reference not available, skipping ONNX model verification")


if __name__ == "__main__":
    main()