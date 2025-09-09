#!/usr/bin/env python3
# Generates matmulinteger.onnx in the same directory and sanity-checks with ReferenceEvaluator.

import os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

HERE = Path(__file__).parent.resolve()
OUT = HERE / "matmulinteger.onnx"

def main():
    # ---- ValueInfos ----
    A  = helper.make_tensor_value_info("A", TensorProto.UINT8, [2, 4])
    B  = helper.make_tensor_value_info("B", TensorProto.UINT8, [4, 3])
    YA = helper.make_tensor_value_info("YA", TensorProto.INT32, [2, 3])

    C  = helper.make_tensor_value_info("C", TensorProto.UINT8, [2, 4])
    D  = helper.make_tensor_value_info("D", TensorProto.UINT8, [4, 3])
    YB = helper.make_tensor_value_info("YB", TensorProto.INT32, [2, 3])

    E  = helper.make_tensor_value_info("E", TensorProto.INT8,  [2, 4])
    F  = helper.make_tensor_value_info("F", TensorProto.UINT8, [4, 2])
    YC = helper.make_tensor_value_info("YC", TensorProto.INT32, [2, 2])

    # ---- ZPs as Constant(Int32) + Cast (no 8-bit initializers) ----
    a0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="a0_i32")
    b0_i32 = numpy_helper.from_array(np.array([0], dtype=np.int32), name="b0_i32")
    a2_i32 = numpy_helper.from_array(np.array([2], dtype=np.int32), name="a2_i32")
    b3_i32 = numpy_helper.from_array(np.array([3], dtype=np.int32), name="b3_i32")

    const_a0 = helper.make_node("Constant", [], ["a0_i32_out"], value=a0_i32)
    const_b0 = helper.make_node("Constant", [], ["b0_i32_out"], value=b0_i32)
    const_a2 = helper.make_node("Constant", [], ["a2_i32_out"], value=a2_i32)
    const_b3 = helper.make_node("Constant", [], ["b3_i32_out"], value=b3_i32)

    cast_a0_u8 = helper.make_node("Cast", ["a0_i32_out"], ["a0_u8"], to=TensorProto.UINT8)
    cast_b0_u8 = helper.make_node("Cast", ["b0_i32_out"], ["b0_u8"], to=TensorProto.UINT8)
    cast_a2_u8 = helper.make_node("Cast", ["a2_i32_out"], ["a2_u8"], to=TensorProto.UINT8)
    cast_b3_u8 = helper.make_node("Cast", ["b3_i32_out"], ["b3_u8"], to=TensorProto.UINT8)
    cast_a0_i8 = helper.make_node("Cast", ["a0_i32_out"], ["a0_i8"], to=TensorProto.INT8)

    # ---- MatMulInteger nodes ----
    nA = helper.make_node("MatMulInteger", ["A","B","a0_u8","b0_u8"], ["YA"], name="mmi_u8u8_zp0")
    nB = helper.make_node("MatMulInteger", ["C","D","a2_u8","b3_u8"], ["YB"], name="mmi_u8u8_zp_const")
    nC = helper.make_node("MatMulInteger", ["E","F","a0_i8","b0_u8"], ["YC"], name="mmi_i8u8_zp0")

    graph = helper.make_graph(
        [
            const_a0,const_b0,const_a2,const_b3,
            cast_a0_u8,cast_b0_u8,cast_a2_u8,cast_b3_u8,cast_a0_i8,
            nA,nB,nC
        ],
        "MatMulIntegerBundle",
        [A,B,C,D,E,F],
        [YA,YB,YC],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    onnx.save(model, OUT.as_posix())
    print(f"Wrote {OUT.name}")

    # ---- Sanity check with ReferenceEvaluator ----
    ref = ReferenceEvaluator(model)

    A_np = np.array([[1,2,3,4],[10,20,30,40]], dtype=np.uint8)
    B_np = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.uint8)
    got = ref.run(None, {"A": A_np, "B": B_np})
    YA_np = A_np.astype(np.int32) @ B_np.astype(np.int32)
    print("YA ok:", np.array_equal(got[0], YA_np))

    C_np = A_np; D_np = B_np
    got = ref.run(None, {"C": C_np, "D": D_np})
    YB_np = (C_np.astype(np.int32)-2) @ (D_np.astype(np.int32)-3)
    print("YB ok:", np.array_equal(got[0], YB_np))

    E_np = np.array([[1,-1,2,-2],[3,-3,4,-4]], dtype=np.int8)
    F_np = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.uint8)
    got = ref.run(None, {"E": E_np, "F": F_np})
    YC_np = E_np.astype(np.int32) @ F_np.astype(np.int32)
    print("YC ok:", np.array_equal(got[0], YC_np))

if __name__ == "__main__":
    main()