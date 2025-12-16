// Import the shared macro (same as matmul / maxpool)
use crate::include_models;
// The names must match your ONNX file stems in tests/matmulinteger/
include_models!(matmulinteger, matmulinteger_ranks);
// These two tests validate that our ONNX files convert end-to-end into a Burn record.
// (Numerical assertions can be added later once we're happy with shapes & codegen.)

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    // Helper to create i32 tensor from nested arrays
    // The ONNX MatMulInteger model uses I32 constants, so we need i32 tensors.
    // We use from_data_dtype to preserve I32 dtype (from_data converts to backend's IntElem which is i64).
    fn tensor_2d_i32<const R: usize, const C: usize>(
        data: [[i32; C]; R],
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 2, Int> {
        let tensor_data = TensorData::from(data);
        Tensor::from_data_dtype(tensor_data, device, DType::I32)
    }

    fn tensor_3d_i32<const B: usize, const R: usize, const C: usize>(
        data: [[[i32; C]; R]; B],
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3, Int> {
        let tensor_data = TensorData::from(data);
        Tensor::from_data_dtype(tensor_data, device, DType::I32)
    }

    fn tensor_1d_i32<const N: usize>(
        data: [i32; N],
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 1, Int> {
        let tensor_data = TensorData::from(data);
        Tensor::from_data_dtype(tensor_data, device, DType::I32)
    }

    // Simple no-zero-point case: check integer matmul → int32 result
    #[test]
    fn matmulinteger_basic() {
        let device = Default::default();
        let model: matmulinteger::Model<TestBackend> = matmulinteger::Model::default();

        // Build inputs matching Python test in matmulinteger.py
        // Note: The ONNX model uses I32 dtype, so we use i32 tensors to match
        // A and B for first MatMulInteger (zero-points: a0=0, b0=0)
        let a = tensor_2d_i32([[1, 2, 3, 4], [10, 20, 30, 40]], &device);
        let b = tensor_2d_i32([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);

        // C and D for second MatMulInteger (zero-points: a2=2, b3=3)
        // Python uses same values as A and B for C and D
        let c = tensor_2d_i32([[1, 2, 3, 4], [10, 20, 30, 40]], &device);
        let d = tensor_2d_i32([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], &device);

        // E and F for third MatMulInteger (zero-points: a0=0, b0=0)
        let e = tensor_2d_i32([[1, -1, 2, -2], [3, -3, 4, -4]], &device);
        let f = tensor_2d_i32([[1, 2], [3, 4], [5, 6], [7, 8]], &device);

        // Forward now takes 6 args and returns 3 outputs (YA, YB, YC)
        let (ya, yb, yc) = model.forward(a, b, c, d, e, f);

        // NdArray backend with i32: Int => i32
        // YA: Computes (A - 0) @ (B - 0) since a0=0 and b0=0
        let expected_ya = TensorData::from([[70i32, 80, 90], [700, 800, 900]]);
        ya.to_data().assert_eq(&expected_ya, true);

        // YB: Computes (C - a2) @ (D - b3) where a2=2 and b3=3 (from ONNX constant nodes)
        let expected_yb = TensorData::from([[20i32, 22, 24], [380, 472, 564]]);
        yb.to_data().assert_eq(&expected_yb, true);

        // YC: Computes (E - 0) @ (F - 0) with mixed signed/unsigned types
        // Since a0=0 and b0=0, the result is just E @ F
        let expected_yc = TensorData::from([[-6i32, -6], [-14, -14]]);
        yc.to_data().assert_eq(&expected_yc, true);
    }

    // Rank/broadcast shapes: mirror your matmul_ranks style but with integer inputs
    #[test]
    fn matmulinteger_ranks() {
        let device = Default::default();
        let model: matmulinteger_ranks::Model<TestBackend> = matmulinteger_ranks::Model::default();

        // Create inputs matching Python test shapes using i32 (ONNX model uses I32)
        // mat2d: [3, 4]
        let mat2d = tensor_2d_i32([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], &device);

        // mat3d: [2, 3, 4]
        let mat3d = tensor_3d_i32(
            [
                [[0, 1, 2, 3], [4, 5, 6, 0], [1, 2, 3, 4]],
                [[5, 6, 0, 1], [2, 3, 4, 5], [6, 0, 1, 2]],
            ],
            &device,
        );

        // vec4: [4]
        let vec4 = tensor_1d_i32([1, 2, 3, 4], &device);

        // vec3: [3]
        let vec3 = tensor_1d_i32([1, 2, 3], &device);

        // sq4: [4, 4]
        let sq4 = tensor_2d_i32(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            &device,
        );

        // mat3d_b: [2, 3, 4]
        let mat3d_b = tensor_3d_i32(
            [
                [[0, 1, 2, 3], [4, 0, 1, 2], [3, 4, 0, 1]],
                [[2, 3, 4, 0], [1, 2, 3, 4], [0, 1, 2, 3]],
            ],
            &device,
        );

        // Call forward with 6 inputs and get 5 outputs
        // Note: The model signature expects: (2D, 3D, 1D, 1D, 2D, 3D) inputs
        // And returns: (2D, 1D, 3D, 1D, 2D) outputs
        let (y_2d_1d, y_1d_2d, y_3d_1d, y_1d_3d, y_2d_2d) =
            model.forward(mat2d, mat3d, vec4, vec3, sq4, mat3d_b);

        // Assert expected outputs
        // y_2d_1d: mat2d @ vec4 = [3, 4] @ [4] → [3]
        let expected_2d_1d = TensorData::from([20i32, 60, 100]);
        y_2d_1d.to_data().assert_eq(&expected_2d_1d, true);

        // y_1d_2d: vec4 @ sq4 = [4] @ [4, 4] → [4]
        let expected_1d_2d = TensorData::from([80i32, 90, 100, 110]);
        y_1d_2d.to_data().assert_eq(&expected_1d_2d, true);

        // y_3d_1d: mat3d @ vec4 = [2, 3, 4] @ [4] → [2, 3]
        let expected_3d_1d = TensorData::from([[20i32, 32, 30], [21, 40, 17]]);
        y_3d_1d.to_data().assert_eq(&expected_3d_1d, true);

        // y_1d_3d: vec3 @ mat3d_b = [3] @ [2, 3, 4] → [2, 4]
        let expected_1d_3d = TensorData::from([[17i32, 13, 4, 10], [4, 10, 16, 17]]);
        y_1d_3d.to_data().assert_eq(&expected_1d_3d, true);

        // y_2d_2d: mat2d @ sq4 = [3, 4] @ [4, 4] → [3, 4]
        let expected_2d_2d = TensorData::from([
            [56i32, 62, 68, 74],
            [152, 174, 196, 218],
            [248, 286, 324, 362],
        ]);
        y_2d_2d.to_data().assert_eq(&expected_2d_2d, true);
    }
}
