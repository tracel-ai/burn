use crate::include_models;
include_models!(
    eye_like,
    eye_like_k1,
    eye_like_int,
    eye_like_k_minus1,
    eye_like_float64,
    eye_like_int32,
    eye_like_bool,
    eye_like_large_k,
    eye_like_1x1,
    eye_like_wide,
    eye_like_neg_large_k
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn eye_like_test() {
        // Test for EyeLike operation
        let device = Default::default();
        let model = eye_like::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor (values don't matter for EyeLike, just the shape)
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // Expected output is a 3x3 identity matrix
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_rectangular_test() {
        // Test for EyeLike operation with rectangular matrix
        let device = Default::default();

        // Create a 3x4 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([3, 4], &device);

        // For rectangular matrices, EyeLike should create identity in top-left corner
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            &device,
        );

        // We can use the same model since EyeLike adapts to input shape
        let model = eye_like::Model::<TestBackend>::new(&device);
        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_k1_test() {
        // Test for EyeLike operation with k=1 (upper diagonal)
        let device = Default::default();
        let model = eye_like_k1::Model::<TestBackend>::new(&device);

        // Create a 4x4 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

        // Expected output has ones on the upper diagonal (k=1)
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_int_test() {
        // Test for EyeLike operation with integer output type
        let device = Default::default();
        let model = eye_like_int::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor (values don't matter)
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        let output = model.forward(input);

        // Output should be integer tensor with identity matrix values
        let expected = Tensor::<TestBackend, 2, burn::tensor::Int>::from_ints(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn eye_like_k_minus1_test() {
        // Test for EyeLike operation with k=-1 (lower diagonal)
        let device = Default::default();
        let model = eye_like_k_minus1::Model::<TestBackend>::new(&device);

        // Create a 4x4 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

        // Expected output has ones on the lower diagonal (k=-1)
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_float64_test() {
        // Test for EyeLike operation with Float64 dtype
        let device = Default::default();
        let model = eye_like_float64::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // Expected output is a 3x3 identity matrix in Float64
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_int32_test() {
        // Test for EyeLike operation with Int32 dtype
        let device = Default::default();
        let model = eye_like_int32::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // Expected output is a 3x3 identity matrix as Int32
        let expected = Tensor::<TestBackend, 2, burn::tensor::Int>::from_ints(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            &device,
        );

        let output = model.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn eye_like_bool_test() {
        // Test for EyeLike operation with Bool dtype
        let device = Default::default();
        let model = eye_like_bool::Model::<TestBackend>::new(&device);

        // Create a 3x3 input tensor
        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // Expected output is a 3x3 identity matrix as Bool
        let expected = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_bool(
            [
                [true, false, false],
                [false, true, false],
                [false, false, true],
            ]
            .into(),
            &device,
        );

        let output = model.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn eye_like_large_k_test() {
        // Test EyeLike with k=5, which is beyond the 3x3 matrix bounds
        // Should result in all zeros since diagonal is completely outside matrix
        let device = Default::default();
        let model = eye_like_large_k::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // k=5 means diagonal starts at (0, 5), which is outside a 3x3 matrix
        // Expected: all zeros
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_neg_large_k_test() {
        // Test EyeLike with k=-5, which is beyond the 3x3 matrix bounds (negative)
        // Should result in all zeros since diagonal is completely outside matrix
        let device = Default::default();
        let model = eye_like_neg_large_k::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([3, 3], &device);

        // k=-5 means diagonal starts at (5, 0), which is outside a 3x3 matrix
        // Expected: all zeros
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_1x1_test() {
        // Test EyeLike with smallest possible 2D matrix (1x1)
        let device = Default::default();
        let model = eye_like_1x1::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([1, 1], &device);

        // 1x1 matrix with main diagonal should have single element = 1.0
        let expected = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn eye_like_wide_test() {
        // Test EyeLike with very wide matrix (1x8)
        // Only one element can be on the diagonal
        let device = Default::default();
        let model = eye_like_wide::Model::<TestBackend>::new(&device);

        let input = Tensor::<TestBackend, 2>::zeros([1, 8], &device);

        // 1x8 matrix: only (0,0) can have a 1, rest are 0
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            &device,
        );

        let output = model.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
