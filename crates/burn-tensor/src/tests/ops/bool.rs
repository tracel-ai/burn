#[burn_tensor_testgen::testgen(bool)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_from_float() {
        let tensor1 = TestTensor::<2>::from([[0.0, 43.0, 0.0], [2.0, -4.2, 31.33]]);
        let data_actual = tensor1.bool().into_data();
        let data_expected = TensorData::from([[false, true, false], [true, true, true]]);
        data_actual.assert_eq(&data_expected, false);
    }

    #[test]
    fn test_from_int() {
        let tensor1 = TestTensorInt::<2>::from([[0, 43, 0], [2, -4, 31]]);
        let data_actual = tensor1.bool().into_data();
        let data_expected = TensorData::from([[false, true, false], [true, true, true]]);
        data_actual.assert_eq(&data_expected, false);
    }

    #[test]
    fn test_bool_and() {
        let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
        let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
        let data_actual = tensor1.bool_and(tensor2).into_data();
        let data_expected = TensorData::from([[false, true, false], [false, false, true]]);
        data_expected.assert_eq(&data_actual, false);
    }

    #[test]
    fn test_bool_or() {
        let tensor1 = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);
        let tensor2 = TestTensorBool::<2>::from([[true, true, false], [false, false, true]]);
        let data_actual = tensor1.bool_or(tensor2).into_data();
        let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
        data_expected.assert_eq(&data_actual, false);
    }
}
