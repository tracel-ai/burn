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
        let expected_ya = TensorData::from([[70i64, 80, 90], [700, 800, 900]]);
        ya.to_data().assert_eq(&expected_ya, true);
    }

    // Rank/broadcast shapes: mirror your matmul_ranks style but with integer inputs
    #[test]
    fn matmulinteger_ranks() {
        let device = Default::default();
        let model: matmulinteger_ranks::Model<TestBackend> =
            matmulinteger_ranks::Model::new(&device);
        // or .default() depending on how your codegen emits the constructor

        // 2D @ 1D → 1D
        let mat2d = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2], [3, 4, 5]], &device);
        let vec1d = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3], &device);
        // 1D @ 2D → 1D
        let vec1d_b = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device);
        let mat2d_square = Tensor::<TestBackend, 2, Int>::from_ints(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            &device,
        );

        // 3D shapes similar to your matmul test (adjust to the exact inputs your generated forward expects)
        // If your generated model takes multiple inputs, pass them in order:
        // let (out_2d_1d, out_1d_2d, ...) = model.forward(mat2d, mat3d, vec1d, vec3, mat2d_square, mat3d_for_vec);

        // Minimal single-case run (2D@1D)
        // If your ranks model expects multiple inputs, mirror the matmul ranks test’s signature.
        // Here’s a simple one-call example assuming the ranks model exposes a similar forward:
        // let out_2d_1d = model.forward_2d_1d(mat2d.clone(), vec1d.clone());
        // (If you don’t generate helper variants, skip this or keep compile-only tests.)

        // Quick sanity for 2D@1D (mat2d @ vec1d) = [ (0*1+1*2+2*3)=8, (3*1+4*2+5*3)=26 ]
        // let expected_2d_1d = TensorData::from([8i32, 26]);
        // out_2d_1d.to_data().assert_eq(&expected_2d_1d, true);
    }
}
