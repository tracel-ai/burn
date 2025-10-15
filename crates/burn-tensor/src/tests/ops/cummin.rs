#[burn_tensor_testgen::testgen(cummin)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_cummin_float_dim_0() {
        let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

        let output = tensor.cummin(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3.0, 1.0, 4.0], [2.0, 1.0, 1.0]]), false);
    }

    #[test]
    fn test_cummin_float_dim_1() {
        let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

        let output = tensor.cummin(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3.0, 1.0, 1.0], [2.0, 2.0, 1.0]]), false);
    }

    #[test]
    fn test_cummin_int_dim_0() {
        let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

        let output = tensor.cummin(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3, 1, 4], [2, 1, 1]]), false);
    }

    #[test]
    fn test_cummin_int_dim_1() {
        let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

        let output = tensor.cummin(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3, 1, 1], [2, 2, 1]]), false);
    }

    #[test]
    fn test_cummin_float_3d() {
        let tensor = TestTensor::<3>::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 6.0], [7.0, 8.0]]]);

        let output = tensor.cummin(2);

        output.into_data().assert_eq(
            &TensorData::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 5.0], [7.0, 7.0]]]),
            false,
        );
    }
}
