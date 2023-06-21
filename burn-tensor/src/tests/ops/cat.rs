#[burn_tensor_testgen::testgen(cat)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_cat_ops_2d_dim0() {
        let tensor_1 = TestTensor::from_data([[1.0, 2.0, 3.0]]);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]]);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_cat_ops_2d_dim1() {
        let tensor_1 = TestTensor::from_data([[1.0, 2.0, 3.0]]);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0, 6.0]]);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 1).into_data();

        let data_expected = Data::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_cat_ops_3d() {
        let tensor_1 = TestTensor::from_data([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]]);
        let tensor_2 = TestTensor::from_data([[[4.0, 5.0, 6.0]]]);

        let data_actual = TestTensor::cat(vec![tensor_1, tensor_2], 0).into_data();

        let data_expected = Data::from([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0]]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
