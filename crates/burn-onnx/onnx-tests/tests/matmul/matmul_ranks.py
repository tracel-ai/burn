#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/matmul/matmul_ranks.onnx
# Tests various rank combinations for matmul broadcasting

import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator
import numpy as np


def main():
    """
    Create an ONNX model with matmul operations of various rank combinations.
    
    Test cases:
    1. 2D × 1D (matrix-vector)
    2. 1D × 2D (vector-matrix)
    3. 3D × 1D (batch matrix-vector)
    4. 1D × 3D (vector-batch matrix)
    5. 2D × 2D (standard matrix multiplication)
    """
    
    # Define inputs
    mat2d_input = helper.make_tensor_value_info("mat2d", TensorProto.FLOAT, [3, 4])
    mat3d_input = helper.make_tensor_value_info("mat3d", TensorProto.FLOAT, [2, 3, 4])
    vec_input = helper.make_tensor_value_info("vec", TensorProto.FLOAT, [4])
    mat2d_square_input = helper.make_tensor_value_info("mat2d_square", TensorProto.FLOAT, [4, 4])
    
    # Test case 1: 2D × 1D (matrix-vector)
    matmul_2d_1d = helper.make_node(
        "MatMul",
        inputs=["mat2d", "vec"],
        outputs=["output_2d_1d"],
        name="matmul_2d_1d"
    )
    
    # Test case 2: 1D × 2D (vector-matrix)
    matmul_1d_2d = helper.make_node(
        "MatMul",
        inputs=["vec", "mat2d_square"],
        outputs=["output_1d_2d"],
        name="matmul_1d_2d"
    )
    
    # Test case 3: 3D × 1D (batch matrix-vector)
    matmul_3d_1d = helper.make_node(
        "MatMul",
        inputs=["mat3d", "vec"],
        outputs=["output_3d_1d"],
        name="matmul_3d_1d"
    )
    
    # Test case 4: 1D × 3D (vector-batch matrix)
    # We need a vector that matches the row dimension of the 3D matrix
    vec3_input = helper.make_tensor_value_info("vec3", TensorProto.FLOAT, [3])
    mat3d_for_vec_input = helper.make_tensor_value_info("mat3d_for_vec", TensorProto.FLOAT, [2, 3, 4])
    
    matmul_1d_3d = helper.make_node(
        "MatMul",
        inputs=["vec3", "mat3d_for_vec"],
        outputs=["output_1d_3d"],
        name="matmul_1d_3d"
    )
    
    # Test case 5: 2D × 2D (standard)
    matmul_2d_2d = helper.make_node(
        "MatMul",
        inputs=["mat2d", "mat2d_square"],
        outputs=["output_2d_2d"],
        name="matmul_2d_2d"
    )
    
    # Define outputs
    output_2d_1d = helper.make_tensor_value_info("output_2d_1d", TensorProto.FLOAT, [3])
    output_1d_2d = helper.make_tensor_value_info("output_1d_2d", TensorProto.FLOAT, [4])
    output_3d_1d = helper.make_tensor_value_info("output_3d_1d", TensorProto.FLOAT, [2, 3])
    output_1d_3d = helper.make_tensor_value_info("output_1d_3d", TensorProto.FLOAT, [2, 4])
    output_2d_2d = helper.make_tensor_value_info("output_2d_2d", TensorProto.FLOAT, [3, 4])
    
    # Create the graph
    graph = helper.make_graph(
        [matmul_2d_1d, matmul_1d_2d, matmul_3d_1d, matmul_1d_3d, matmul_2d_2d],
        "matmul_ranks_model",
        [mat2d_input, mat3d_input, vec_input, vec3_input, mat2d_square_input, mat3d_for_vec_input],
        [output_2d_1d, output_1d_2d, output_3d_1d, output_1d_3d, output_2d_2d],
    )
    
    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    
    # Save the model
    file_name = "matmul_ranks.onnx"
    onnx.save(model, file_name)
    print(f"Finished exporting model to {file_name}")
    
    # Compute expected outputs using ReferenceEvaluator
    print("\nComputing expected outputs using ReferenceEvaluator:")
    
    # Create test inputs
    np.random.seed(42)  # For reproducibility
    mat2d = np.arange(12, dtype=np.float32).reshape(3, 4)
    mat3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    vec = np.arange(4, dtype=np.float32)
    vec3 = np.arange(3, dtype=np.float32)
    mat2d_square = np.arange(16, dtype=np.float32).reshape(4, 4)
    mat3d_for_vec = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    
    # Use ReferenceEvaluator to compute outputs
    sess = ReferenceEvaluator(model)
    outputs = sess.run(None, {
        "mat2d": mat2d,
        "mat3d": mat3d,
        "vec": vec,
        "vec3": vec3,
        "mat2d_square": mat2d_square,
        "mat3d_for_vec": mat3d_for_vec
    })
    
    print(f"\nTest inputs:")
    print(f"  mat2d shape: {mat2d.shape}")
    print(f"  mat3d shape: {mat3d.shape}")
    print(f"  vec shape: {vec.shape}")
    print(f"  vec3 shape: {vec3.shape}")
    print(f"  mat2d_square shape: {mat2d_square.shape}")
    
    print(f"\nExpected outputs:")
    print(f"  output_2d_1d (mat2d @ vec): {outputs[0]}")
    print(f"  output_1d_2d (vec @ mat2d_square): {outputs[1]}")
    print(f"  output_3d_1d (mat3d @ vec): {outputs[2]}")
    print(f"  output_1d_3d (vec3 @ mat3d_for_vec): {outputs[3]}")
    print(f"  output_2d_2d (mat2d @ mat2d_square): {outputs[4]}")
    
    print(f"\nFor Rust tests:")
    print(f"  output_2d_1d: {outputs[0].tolist()}")
    print(f"  output_1d_2d: {outputs[1].tolist()}")
    print(f"  output_3d_1d: {outputs[2].tolist()}")
    print(f"  output_1d_3d: {outputs[3].tolist()}")
    print(f"  output_2d_2d: {outputs[4].tolist()}")


if __name__ == "__main__":
    main()