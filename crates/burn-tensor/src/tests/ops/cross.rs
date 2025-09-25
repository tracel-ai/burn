#[burn_tensor_testgen::testgen(cross)]
mod tests {
    use super::*;
    use burn_tensor::might_panic;
    use burn_tensor::{Tensor, TensorData, backend::Backend, s};

    #[test]
    fn test_cross_3d_last_dim() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_non_contiguous_last_dim() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
        let tensor_2 = TestTensor::from([[4.0, 3.0], [-2.0, 5.0], [1.0, -2.0]]);

        let output = tensor_1.cross(tensor_2.permute([1, 0]), -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
            false,
        );
    }

    #[cfg(feature = "std")]
    #[might_panic(reason = "not implemented: Cross product on non-last dimension")]
    #[test]
    fn test_cross_3d_dim0() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
        let tensor_2 = TestTensor::from([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

        let output = tensor_1.cross(tensor_2, 0);

        output.into_data().assert_eq(
            &TensorData::from([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_3d_broadcast() {
        let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0]]);
        let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[-7.0, -21.0, -14.0], [19.0, -13.0, -4.0]]),
            false,
        );
    }

    #[test]
    fn test_cross_4d_last_dim() {
        let tensor_1 = TestTensor::<3>::from([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let tensor_2 = TestTensor::from([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]);

        let output = tensor_1.cross(tensor_2, -1);

        output.into_data().assert_eq(
            &TensorData::from([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]),
            false,
        );
    }

    // Helper to compute expected cross product for 2-D (N × 3) tensors.
    fn manual_cross(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<[f32; 3]> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                [
                    x[1] * y[2] - x[2] * y[1],
                    x[2] * y[0] - x[0] * y[2],
                    x[0] * y[1] - x[1] * y[0],
                ]
            })
            .collect()
    }

    #[test]
    fn forward_matches_manual_cross() {
        let a_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_raw = [[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]];
        let a = TestTensor::<2>::from(a_raw);
        let b = TestTensor::<2>::from(b_raw);

        let out = a.cross(b.clone(), 1);
        let expected_vec = manual_cross(&a_raw, &b_raw);
        let expected: [[f32; 3]; 2] = [expected_vec[0], expected_vec[1]];

        out.into_data()
            .assert_eq(&TensorData::from(expected), false);
    }
}
