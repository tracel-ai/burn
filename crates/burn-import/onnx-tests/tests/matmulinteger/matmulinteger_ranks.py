#!/usr/bin/env python3
# Generates matmulinteger_ranks.onnx in the same directory and sanity-checks with ReferenceEvaluator.

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = Path(__file__).parent.resolve()
OUT = HERE / "matmulinteger_ranks.onnx"

def main():
    # Inputs (uint8) analogous to MatMul ranks
    mat2d   = helper.make_tensor_value_info("mat2d",   TensorProto.UINT8, [3, 4])
    mat3d   = helper.make_tensor_value_info("mat3d",   TensorProto.UINT8, [2, 3, 4])
    vec4    = helper.make_tensor_value_info("vec4",    TensorProto.UINT8, [4])
    vec3    = helper.make_tensor_value_info("vec3",    TensorProto.UINT8, [3])
    sq4     = helper.make_tensor_value_info("sq4",     TensorProto.UINT8, [4, 4])
    mat3d_b = helper.make_tensor_value_info("mat3d_b", TensorProto.UINT8, [2, 3, 4])

    # Outputs (int32)
    y_2d_1d = helper.make_tensor_value_info("y_2d_1d", TensorProto.INT32, [3])
    y_1d_2d = helper.make_tensor_value_info("y_1d_2d", TensorProto.INT32, [4])
    y_3d_1d = helper.make_tensor_value_info("y_3d_1d", TensorProto.INT32, [2,3])
    y_1d_3d = helper.make_tensor_value_info("y_1d_3d", TensorProto.INT32, [2,4])
    y_2d_2d = helper.make_tensor_value_info("y_2d_2d", TensorProto.INT32, [3,4])

    # ZPs as Constant(Int32) + Cast (shared)
    a0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="a0_i32")
    b0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="b0_i32")
    const_a0 = helper.make_node("Constant", [], ["a0_i32_out"], value=a0_i32)
    const_b0 = helper.make_node("Constant", [], ["b0_i32_out"], value=b0_i32)
    cast_a0_u8 = helper.make_node("Cast", ["a0_i32_out"], ["a0_u8"], to=TensorProto.UINT8)
    cast_b0_u8 = helper.make_node("Cast", ["b0_i32_out"], ["b0_u8"], to=TensorProto.UINT8)

    # Rank cases (zp=0 for all)
    n_2d_1d = helper.make_node("MatMulInteger", ["mat2d",  "vec4",   "a0_u8","b0_u8"], ["y_2d_1d"], name="mmi_2d_1d")
    n_1d_2d = helper.make_node("MatMulInteger", ["vec4",   "sq4",    "a0_u8","b0_u8"], ["y_1d_2d"], name="mmi_1d_2d")
    n_3d_1d = helper.make_node("MatMulInteger", ["mat3d",  "vec4",   "a0_u8","b0_u8"], ["y_3d_1d"], name="mmi_3d_1d")
    n_1d_3d = helper.make_node("MatMulInteger", ["vec3",   "mat3d_b","a0_u8","b0_u8"], ["y_1d_3d"], name="mmi_1d_3d")
    n_2d_2d = helper.make_node("MatMulInteger", ["mat2d",  "sq4",    "a0_u8","b0_u8"], ["y_2d_2d"], name="mmi_2d_2d")

    graph = helper.make_graph(
        [const_a0,const_b0,cast_a0_u8,cast_b0_u8,
         n_2d_1d,n_1d_2d,n_3d_1d,n_1d_3d,n_2d_2d],
        "MatMulIntegerRanks",
        [mat2d,mat3d,vec4,vec3,sq4,mat3d_b],
        [y_2d_1d,y_1d_2d,y_3d_1d,y_1d_3d,y_2d_2d],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, OUT.as_posix())
    print(f"Wrote {OUT.name}")

    # ---- Sanity with ReferenceEvaluator (deterministic small inputs) ----
    ref = ReferenceEvaluator(model)

    mat2d_np   = np.arange(12, dtype=np.uint8).reshape(3,4)
    mat3d_np   = (np.arange(24, dtype=np.uint8).reshape(2,3,4) % 7).astype(np.uint8)
    vec4_np    = np.array([1,2,3,4], dtype=np.uint8)
    vec3_np    = np.array([1,2,3], dtype=np.uint8)
    sq4_np     = np.arange(16, dtype=np.uint8).reshape(4,4)
    mat3d_b_np = (np.arange(24, dtype=np.uint8).reshape(2,3,4) % 5).astype(np.uint8)

    print("\n" + "="*60)
    print("Test data for matmulinteger_ranks.onnx")
    print("="*60)
    
    # Print test inputs
    print("\nTest input mat2d shape:", mat2d_np.shape)
    print("Test input mat2d:", mat2d_np.tolist())
    
    print("\nTest input mat3d shape:", mat3d_np.shape)
    print("Test input mat3d:", mat3d_np.tolist())
    
    print("\nTest input vec4 shape:", vec4_np.shape)
    print("Test input vec4:", vec4_np.tolist())
    
    print("\nTest input vec3 shape:", vec3_np.shape)
    print("Test input vec3:", vec3_np.tolist())
    
    print("\nTest input sq4 shape:", sq4_np.shape)
    print("Test input sq4:", sq4_np.tolist())
    
    print("\nTest input mat3d_b shape:", mat3d_b_np.shape)
    print("Test input mat3d_b:", mat3d_b_np.tolist())

    # Helper: int32 matmul with broadcasting semantics where applicable
    def mm(a, b):
        return a.astype(np.int32) @ b.astype(np.int32)

    # Compute expected outputs
    y_2d_1d_np = mm(mat2d_np, vec4_np)
    y_1d_2d_np = mm(vec4_np, sq4_np)
    y_3d_1d_np = np.matmul(mat3d_np.astype(np.int32), vec4_np.astype(np.int32))  # (2,3)
    y_1d_3d_np = np.matmul(vec3_np.astype(np.int32),  mat3d_b_np.astype(np.int32))  # (2,4)
    y_2d_2d_np = mm(mat2d_np, sq4_np)

    # Run model inference
    got_all = ref.run(None, {
        "mat2d": mat2d_np,
        "mat3d": mat3d_np,
        "vec4":  vec4_np,
        "vec3":  vec3_np,
        "sq4":   sq4_np,
        "mat3d_b": mat3d_b_np,
    })

    # Test y_2d_1d: mat2d @ vec4 = [3, 4] @ [4] → [3]
    print("\nTest y_2d_1d = mat2d @ vec4 (zero-points: a0=0, b0=0)")
    print("Expected y_2d_1d shape:", y_2d_1d_np.shape)
    print("Expected y_2d_1d:", y_2d_1d_np.tolist())
    print("y_2d_1d verification:", "PASS" if np.array_equal(got_all[0], y_2d_1d_np) else "FAIL")
    
    # Test y_1d_2d: vec4 @ sq4 = [4] @ [4, 4] → [4]
    print("\nTest y_1d_2d = vec4 @ sq4 (zero-points: a0=0, b0=0)")
    print("Expected y_1d_2d shape:", y_1d_2d_np.shape)
    print("Expected y_1d_2d:", y_1d_2d_np.tolist())
    print("y_1d_2d verification:", "PASS" if np.array_equal(got_all[1], y_1d_2d_np) else "FAIL")
    
    # Test y_3d_1d: mat3d @ vec4 = [2, 3, 4] @ [4] → [2, 3]
    print("\nTest y_3d_1d = mat3d @ vec4 (zero-points: a0=0, b0=0)")
    print("Expected y_3d_1d shape:", y_3d_1d_np.shape)
    print("Expected y_3d_1d:", y_3d_1d_np.tolist())
    print("y_3d_1d verification:", "PASS" if np.array_equal(got_all[2], y_3d_1d_np) else "FAIL")
    
    # Test y_1d_3d: vec3 @ mat3d_b = [3] @ [2, 3, 4] → [2, 4]
    print("\nTest y_1d_3d = vec3 @ mat3d_b (zero-points: a0=0, b0=0)")
    print("Expected y_1d_3d shape:", y_1d_3d_np.shape)
    print("Expected y_1d_3d:", y_1d_3d_np.tolist())
    print("y_1d_3d verification:", "PASS" if np.array_equal(got_all[3], y_1d_3d_np) else "FAIL")
    
    # Test y_2d_2d: mat2d @ sq4 = [3, 4] @ [4, 4] → [3, 4]
    print("\nTest y_2d_2d = mat2d @ sq4 (zero-points: a0=0, b0=0)")
    print("Expected y_2d_2d shape:", y_2d_2d_np.shape)
    print("Expected y_2d_2d:", y_2d_2d_np.tolist())
    print("y_2d_2d verification:", "PASS" if np.array_equal(got_all[4], y_2d_2d_np) else "FAIL")

if __name__ == "__main__":
    main()