#[burn_tensor_testgen::testgen(any)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_any() {
        // test float tensor
        let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([true]);
        data_expected.assert_eq(&data_actual, false);

        let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([false]);
        data_expected.assert_eq(&data_actual, false);

        // test int tensor
        let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([true]);
        data_expected.assert_eq(&data_actual, false);

        let tensor = TestTensorInt::<2>::from([[0, 0, 0], [0, 0, 0]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([false]);
        data_expected.assert_eq(&data_actual, false);

        // test bool tensor
        let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([true]);
        data_expected.assert_eq(&data_actual, false);

        let tensor = TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);
        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([false]);
        data_expected.assert_eq(&data_actual, false);
    }

    #[test]
    fn test_any_dim() {
        let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);
        let data_actual = tensor.any_dim(1).into_data();
        let data_expected = TensorData::from([[false], [true]]);
        data_expected.assert_eq(&data_actual, false);

        // test int tensor
        let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
        let data_actual = tensor.any_dim(1).into_data();
        let data_expected = TensorData::from([[false], [true]]);
        data_expected.assert_eq(&data_actual, false);

        // test bool tensor
        let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
        let data_actual = tensor.any_dim(1).into_data();
        let data_expected = TensorData::from([[false], [true]]);
        data_expected.assert_eq(&data_actual, false);
    }
}
