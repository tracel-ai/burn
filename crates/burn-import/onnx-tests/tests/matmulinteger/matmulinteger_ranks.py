#!/usr/bin/env python3
# outputs: onnx-tests/tests/matmulinteger/matmulinteger_ranks.onnx
import os, onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper

def main():
    os.makedirs("onnx-tests/tests/matmulinteger", exist_ok=True)

    # Inputs (uint8) analogous to matmul_ranks.py
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

    # ZPs as Constant(Int32) + Cast (shared by all nodes)
    a0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="a0_i32")
    b0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="b0_i32")
    const_a0 = helper.make_node("Constant", [], ["a0_i32_out"], value=a0_i32)
    const_b0 = helper.make_node("Constant", [], ["b0_i32_out"], value=b0_i32)
    cast_a0_u8 = helper.make_node("Cast", ["a0_i32_out"], ["a0_u8"], to=TensorProto.UINT8)
    cast_b0_u8 = helper.make_node("Cast", ["b0_i32_out"], ["b0_u8"], to=TensorProto.UINT8)

    # Rank cases (all zp=0)
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
    out = "onnx-tests/tests/matmulinteger/matmulinteger_ranks.onnx"
    onnx.save(model, out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()