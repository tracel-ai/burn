#!/usr/bin/env python3
# outputs: onnx-tests/tests/matmulinteger/matmulinteger.onnx
import os, onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper

def main():
    os.makedirs("onnx-tests/tests/matmulinteger", exist_ok=True)

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

    # ---- ZPs as Constant(Int32) + Cast (NO 8-bit initializers) ----
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
        # NOTE: no initializer list at all
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    out = "onnx-tests/tests/matmulinteger/matmulinteger.onnx"
    onnx.save(model, out)
    print(f"Wrote {out}")

    # Optional: expected outputs (quick sanity)
    A_np = np.array([[1,2,3,4],[10,20,30,40]], dtype=np.uint8)
    B_np = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.uint8)
    YA_np = A_np.astype(np.int32) @ B_np.astype(np.int32)
    print("Expected YA:", YA_np.tolist())

    C_np = A_np; D_np = B_np
    YB_np = (C_np.astype(np.int32)-2) @ (D_np.astype(np.int32)-3)
    print("Expected YB:", YB_np.tolist())

    E_np = np.array([[1,-1,2,-2],[3,-3,4,-4]], dtype=np.int8)
    F_np = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.uint8)
    YC_np = E_np.astype(np.int32) @ F_np.astype(np.int32)
    print("Expected YC:", YC_np.tolist())

if __name__ == "__main__":
    main()