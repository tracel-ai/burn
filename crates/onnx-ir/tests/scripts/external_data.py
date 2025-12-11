#!/usr/bin/env python3
"""Generate ONNX test models with external data storage.

This script creates ONNX models that store tensor weights in external files,
which is used for models >2GB that exceed protobuf's size limit.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import os

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def create_external_data_model():
    """Create a simple model with weights stored externally.

    The model: Y = X * weight + bias
    - weight is a [4, 4] float32 tensor stored externally
    - bias is a [4] float32 tensor stored externally
    """
    # Create input
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # Create weight and bias tensors with known values for testing
    weight_data = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ], dtype=np.float32)

    bias_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    weight_tensor = numpy_helper.from_array(weight_data, name="weight")
    bias_tensor = numpy_helper.from_array(bias_data, name="bias")

    # Create nodes
    matmul_node = helper.make_node("MatMul", ["X", "weight"], ["matmul_out"], name="matmul")
    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["Y"], name="add")

    # Create graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        "external_data_test",
        [X],
        [Y],
        [weight_tensor, bias_tensor],
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # Save with external data
    output_path = os.path.join(FIXTURES_DIR, "external_data.onnx")
    external_data_path = "external_data.bin"

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path,
        size_threshold=0,  # Store all tensors externally
    )

    print(f"Created: {output_path}")
    print(f"External data: {os.path.join(FIXTURES_DIR, external_data_path)}")

    # Verify the model loads correctly
    loaded = onnx.load(output_path)
    onnx.checker.check_model(loaded)
    print("Model validation passed!")


def create_external_data_with_offset():
    """Create a model with external data at non-zero offset.

    This tests the offset field in external_data.
    """
    # Create a simple model: Y = X + const
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    # Create constant tensor
    const_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    const_tensor = numpy_helper.from_array(const_data, name="const")

    add_node = helper.make_node("Add", ["X", "const"], ["Y"], name="add")

    graph = helper.make_graph(
        [add_node],
        "external_data_offset_test",
        [X],
        [Y],
        [const_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    output_path = os.path.join(FIXTURES_DIR, "external_data_offset.onnx")
    external_data_path = "external_data_offset.bin"

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path,
        size_threshold=0,
    )

    print(f"Created: {output_path}")


def create_mixed_data_model():
    """Create a model with both embedded and external data.

    Small tensors stay embedded, large tensor goes external.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

    # Large weight tensor (will be external)
    weight_data = np.random.randn(64, 64).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight_data, name="weight")

    # Small bias tensor (will stay embedded due to size_threshold)
    bias_data = np.zeros(64, dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias_data, name="bias")

    matmul_node = helper.make_node("MatMul", ["X", "weight"], ["matmul_out"], name="matmul")
    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["Y"], name="add")

    graph = helper.make_graph(
        [matmul_node, add_node],
        "mixed_data_test",
        [X],
        [Y],
        [weight_tensor, bias_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    output_path = os.path.join(FIXTURES_DIR, "mixed_data.onnx")
    external_data_path = "mixed_data.bin"

    # Only externalize tensors > 1KB (weight is 64*64*4 = 16KB, bias is 64*4 = 256B)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path,
        size_threshold=1024,  # 1KB threshold
    )

    print(f"Created: {output_path}")


if __name__ == "__main__":
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    print("Generating external data test models...\n")

    create_external_data_model()
    print()

    create_external_data_with_offset()
    print()

    create_mixed_data_model()
    print()

    print("Done!")
