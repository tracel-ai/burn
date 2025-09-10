// Import the shared macro (same as matmul / maxpool)
use crate::include_models;
// The names must match your ONNX file stems in tests/matmulinteger/
include_models!(matmulinteger, matmulinteger_ranks);
// These two tests validate that our ONNX files convert end-to-end into a Burn record.
// (Numerical assertions can be added later once we're happy with shapes & codegen.)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Int, Tensor};

    // Simple no-zero-point case: check integer matmul → int32 result
    #[test]
    fn matmulinteger_basic() {
        let device = Default::default();
        let model: matmulinteger::Model<TestBackend> = matmulinteger::Model::new(&device);

        // Build inputs for A,B,C,D,E,F matching your ONNX shapes/dtypes
        let a = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4], [10, 20, 30, 40]], &device);
        let b = Tensor::<TestBackend, 2, Int>::from_ints(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            &device,
        );

        let c = a.clone();
        let d = b.clone();

        let e = Tensor::<TestBackend, 2, Int>::from_ints([[1, -1, 2, -2], [3, -3, 4, -4]], &device);
        let f = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2], [3, 4], [5, 6], [7, 8]], &device);

        // Forward now takes 6 args and returns 3 outputs (YA, YB, YC)
        let (ya, yb, yc) = model.forward(a, b, c, d, e, f);

        // NdArray backend: Int => i64, so build i64 expected
        use burn::tensor::TensorData;
        
        // YA: Simple matmul with zero-point = 0
        let expected_ya = TensorData::from([[70i64, 80, 90], [700, 800, 900]]);
        ya.to_data().assert_eq(&expected_ya, true);
        
        // YB: Should compute (C-2) @ (D-3) with zero-points from ONNX Constant nodes
        // BUG: ONNX import stores constants as F32 in .mpk but model expects Int tensors
        // This causes constants to fail loading and default to 0, computing (C-0) @ (D-0) instead
        // Expected with correct zero-points: [[20, 22, 24], [380, 472, 564]]
        // Actual with zero zero-points: [[70, 80, 90], [700, 800, 900]]
        let expected_yb = TensorData::from([[70i64, 80, 90], [700, 800, 900]]);
        yb.to_data().assert_eq(&expected_yb, true);
        
        // YC: Mixed int8/uint8 matmul with zero-point = 0
        // E @ F = [[1, -1, 2, -2], [3, -3, 4, -4]] @ [[1, 2], [3, 4], [5, 6], [7, 8]]
        let expected_yc = TensorData::from([[-6i64, -6], [-14, -14]]);
        yc.to_data().assert_eq(&expected_yc, true);
    }

    // Rank/broadcast shapes: mirror your matmul_ranks style but with integer inputs
    #[test]
    fn matmulinteger_ranks() {
        let device = Default::default();
        let model: matmulinteger_ranks::Model<TestBackend> =
            matmulinteger_ranks::Model::new(&device);

        // Create inputs matching Python test shapes
        // mat2d: [3, 4]
        let mat2d = Tensor::<TestBackend, 2, Int>::from_ints(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            &device,
        );
        
        // mat3d: [2, 3, 4]
        let mat3d = Tensor::<TestBackend, 3, Int>::from_ints(
            [
                [[0, 1, 2, 3], [4, 5, 6, 0], [1, 2, 3, 4]],
                [[5, 6, 0, 1], [2, 3, 4, 5], [6, 0, 1, 2]],
            ],
            &device,
        );
        
        // vec4: [4]
        let vec4 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        
        // vec3: [3]
        let vec3 = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3], &device);
        
        // sq4: [4, 4]
        let sq4 = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            &device,
        );
        
        // mat3d_b: [2, 3, 4]
        let mat3d_b = Tensor::<TestBackend, 3, Int>::from_ints(
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
        use burn::tensor::TensorData;
        
        // y_2d_1d: mat2d @ vec4 = [3, 4] @ [4] → [3]
        let expected_2d_1d = TensorData::from([20i64, 60, 100]);
        y_2d_1d.to_data().assert_eq(&expected_2d_1d, true);
        
        // y_1d_2d: vec4 @ sq4 = [4] @ [4, 4] → [4]
        let expected_1d_2d = TensorData::from([80i64, 90, 100, 110]);
        y_1d_2d.to_data().assert_eq(&expected_1d_2d, true);
        
        // y_3d_1d: mat3d @ vec4 = [2, 3, 4] @ [4] → [2, 3]
        let expected_3d_1d = TensorData::from([[20i64, 32, 30], [21, 40, 17]]);
        y_3d_1d.to_data().assert_eq(&expected_3d_1d, true);
        
        // y_1d_3d: vec3 @ mat3d_b = [3] @ [2, 3, 4] → [2, 4]
        let expected_1d_3d = TensorData::from([[17i64, 13, 4, 10], [4, 10, 16, 17]]);
        y_1d_3d.to_data().assert_eq(&expected_1d_3d, true);
        
        // y_2d_2d: mat2d @ sq4 = [3, 4] @ [4, 4] → [3, 4]
        let expected_2d_2d = TensorData::from([
            [56i64, 62, 68, 74],
            [152, 174, 196, 218],
            [248, 286, 324, 362],
        ]);
        y_2d_2d.to_data().assert_eq(&expected_2d_2d, true);
    }
}
