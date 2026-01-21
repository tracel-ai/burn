#!/usr/bin/env python3

# used to generate model: gather_with_shape_indices.onnx

# This test creates the exact scenario that required our runtime Shape indices implementation:
# Using Shape type AS indices in gather operations (not just as data input).

import numpy as np
import onnx
import onnx.helper

try:
    import onnx.reference
    REFERENCE_AVAILABLE = True
except ImportError:
    REFERENCE_AVAILABLE = False
    print("Warning: onnx.reference not available, skipping verification")


def build_model():
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=onnx.helper.make_graph(name="main_graph", nodes=[
            # Create some data to gather from
            onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=["data"],
                name="/Constant1",
                value=onnx.helper.make_tensor(
                    name="data_tensor",
                    data_type=onnx.TensorProto.INT64,
                    dims=[5],
                    vals=[100, 200, 300, 400, 500]
                )
            ),
            # Get shape of input tensor [2, 3]
            onnx.helper.make_node(
                "Shape",
                inputs=["input1"],
                outputs=["shape1"],
                name="/Shape1"
            ),
            # Use shape AS indices for gather - this is the key test case
            # The shape [2, 3] will be used as indices to gather from [100, 200, 300, 400, 500]
            # This should gather elements at indices 2 and 3, giving us [300, 400]
            onnx.helper.make_node(
                "Gather",
                inputs=["data", "shape1"],
                outputs=["output1"],
                name="/Gather",
                axis=0
            ),
        ],
        inputs=[
            onnx.helper.make_value_info(
                name="input1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.FLOAT, shape=[2, 3]
                ),
            ),
        ],
        outputs=[
            onnx.helper.make_value_info(
                name="output1",
                type_proto=onnx.helper.make_tensor_type_proto(
                    elem_type=onnx.TensorProto.INT64, shape=[2]
                ),
            )
        ]),
    )


def verify_model_with_reference(onnx_model):
    """Verify the model behavior using onnx.reference"""
    if not REFERENCE_AVAILABLE:
        print("Skipping verification - onnx.reference not available")
        return
    
    try:
        # Create reference implementation
        ref = onnx.reference.ReferenceEvaluator(onnx_model)
        
        # Test input with shape [2, 3]
        input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Run the model
        result = ref.run(None, {"input1": input_tensor})
        output = result[0]
        
        # Verify the result
        # Shape of input is [2, 3], using this as indices to gather from [100, 200, 300, 400, 500]
        # Should gather at indices [2, 3] giving us [300, 400]
        expected = np.array([300, 400], dtype=np.int64)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Expected output: {expected}")
        print(f"Actual output: {output}")
        print(f"Output type: {type(output)}")
        
        if np.array_equal(output, expected):
            print("✓ Verification PASSED: Model produces expected result")
        else:
            print("✗ Verification FAILED: Model output doesn't match expected result")
            
    except Exception as e:
        print(f"Verification failed with error: {e}")


def main():
    onnx_model = build_model()
    file_name = "gather_with_shape_indices.onnx"

    # Ensure valid ONNX:
    onnx.checker.check_model(onnx_model)
    
    # Verify with reference implementation
    verify_model_with_reference(onnx_model)

    onnx.save(onnx_model, file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":
    main()