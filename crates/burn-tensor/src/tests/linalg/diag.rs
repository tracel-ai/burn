#[burn_tensor_testgen::testgen(diag)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, linalg::diag};

    #[test]
    fn test_diag_2d_square() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        let expected = TestTensor::<1>::from_data([1.0, 4.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_2d_tall() {
        let device = Default::default();
        // 4x2 matrix (tall) - min(4,2) = 2 diagonal elements
        let tensor =
            TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Result should have shape [2] with values [1.0, 4.0]
        let expected = TestTensor::<1>::from_data([1.0, 4.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_2d_wide() {
        let device = Default::default();
        // 2x4 matrix (wide) - min(2,4) = 2 diagonal elements
        let tensor =
            TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Result should have shape [2] with values [1.0, 6.0]
        let expected = TestTensor::<1>::from_data([1.0, 6.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_3d_batch_square() {
        let device = Default::default();
        // Batch of 2 matrices, each 2x2
        let tensor = TestTensor::<3>::from_data(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &device,
        );
        let result = diag::<_, 3, 2, _>(tensor);
        // Result should have shape [2, 2]
        let expected = TestTensor::<2>::from_data([[1.0, 4.0], [5.0, 8.0]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_3d_batch_tall() {
        let device = Default::default();
        // Batch of 2 matrices, each 3x2 (tall)
        let tensor = TestTensor::<3>::from_data(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            &device,
        );
        let result = diag::<_, 3, 2, _>(tensor);
        // Result should have shape [2, 2] - min(3,2) = 2 diagonal elements each
        let expected = TestTensor::<2>::from_data([[1.0, 4.0], [7.0, 10.0]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_3d_batch_wide() {
        let device = Default::default();
        // Batch of 2 matrices, each 2x3 (wide)
        let tensor = TestTensor::<3>::from_data(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device,
        );
        let result = diag::<_, 3, 2, _>(tensor);
        // Result should have shape [2, 2] - min(2,3) = 2 diagonal elements each
        let expected = TestTensor::<2>::from_data([[1.0, 5.0], [7.0, 11.0]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_4d_batch_channel_square() {
        let device = Default::default();
        // [batch=2, channel=2, rows=2, cols=2]
        let tensor = TestTensor::<4>::from_data(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            ],
            &device,
        );
        let result = diag::<_, 4, 3, _>(tensor);
        // Result should have shape [2, 2, 2]
        let expected = TestTensor::<3>::from_data(
            [[[1.0, 4.0], [5.0, 8.0]], [[9.0, 12.0], [13.0, 16.0]]],
            &device,
        );

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_4d_batch_channel_tall() {
        let device = Default::default();
        // [batch=2, channel=1, rows=3, cols=2]
        let tensor = TestTensor::<4>::from_data(
            [
                [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
                [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
            ],
            &device,
        );
        let result = diag::<_, 4, 3, _>(tensor);
        // Result should have shape [2, 1, 2] - min(3,2) = 2 diagonal elements each
        let expected = TestTensor::<3>::from_data([[[1.0, 4.0]], [[7.0, 10.0]]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_4d_batch_channel_wide() {
        let device = Default::default();
        // [batch=1, channel=2, rows=2, cols=4]
        let tensor = TestTensor::<4>::from_data(
            [[
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ]],
            &device,
        );
        let result = diag::<_, 4, 3, _>(tensor);
        // Result should have shape [1, 2, 2] - min(2,4) = 2 diagonal elements each
        let expected = TestTensor::<3>::from_data([[[1.0, 6.0], [9.0, 14.0]]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_1x1() {
        let device = Default::default();
        // Single element matrix
        let tensor = TestTensor::<2>::from_data([[5.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Should return [5.0] with shape [1]
        let expected = TestTensor::<1>::from_data([5.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_single_row() {
        let device = Default::default();
        // Single row matrix
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // min(1,3) = 1, should return [1.0] with shape [1]
        let expected = TestTensor::<1>::from_data([1.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_single_column() {
        let device = Default::default();
        // Single column matrix
        let tensor = TestTensor::<2>::from_data([[1.0], [2.0], [3.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // min(3,1) = 1, should return [1.0] with shape [1]
        let expected = TestTensor::<1>::from_data([1.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_zeros() {
        let device = Default::default();
        // Matrix with zeros on diagonal
        let tensor = TestTensor::<2>::from_data([[0.0, 1.0], [2.0, 0.0]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Should extract diagonal: [0.0, 0.0]
        let expected = TestTensor::<1>::from_data([0.0, 0.0], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_batch_single_element() {
        let device = Default::default();
        // Batch with single element matrices
        let tensor = TestTensor::<3>::from_data([[[5.0]], [[7.0]]], &device);
        let result = diag::<_, 3, 2, _>(tensor);
        // Should return [[5.0], [7.0]] with shape [2, 1]
        let expected = TestTensor::<2>::from_data([[5.0], [7.0]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_batch_mixed_zeros() {
        let device = Default::default();
        // Batch with mixed zero and non-zero diagonal elements
        let tensor = TestTensor::<3>::from_data(
            [[[1.0, 2.0], [3.0, 0.0]], [[0.0, 5.0], [6.0, 7.0]]],
            &device,
        );
        let result = diag::<_, 3, 2, _>(tensor);
        // Should return [[1.0, 0.0], [0.0, 7.0]] with shape [2, 2]
        let expected = TestTensor::<2>::from_data([[1.0, 0.0], [0.0, 7.0]], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_int_tensor() {
        let device = Default::default();
        // Test with integer tensor
        let tensor = TestTensorInt::<2>::from_data([[1, 2], [3, 4]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Result should have shape [2] with values [1, 4]
        let expected = TestTensorInt::<1>::from_data([1, 4], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    fn test_diag_int_3x3() {
        let device = Default::default();
        // Test with 3x3 integer matrix
        let tensor = TestTensorInt::<2>::from_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]], &device);
        let result = diag::<_, 2, 1, _>(tensor);
        // Result should have shape [3] with values [1, 5, 9]
        let expected = TestTensorInt::<1>::from_data([1, 5, 9], &device);

        assert_eq!(result.to_data(), expected.to_data());
    }

    #[test]
    #[should_panic]
    fn test_diag_1d_should_panic() {
        let device = Default::default();
        // 1D tensor should panic - diagonal requires at least 2 dimensions
        let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
        let _result = diag::<_, 1, 0, _>(tensor);
    }

    #[test]
    #[should_panic]
    fn test_diag_wrong_output_rank_should_panic() {
        let device = Default::default();
        // Providing wrong output rank should panic
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let _result = diag::<_, 2, 2, _>(tensor); // Should be 2,1 not 2,2
    }
}
