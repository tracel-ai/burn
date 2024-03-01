#[burn_tensor_testgen::testgen(all_op)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_all() {
        // test float tensor
        let tensor = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([false]);
        assert_eq!(data_expected, data_actual);

        let tensor = TestTensor::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([true]);
        assert_eq!(data_expected, data_actual);

        // test int tensor
        let tensor = TestTensorInt::from([[0, 1, 0], [1, -1, 1]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([false]);
        assert_eq!(data_expected, data_actual);

        let tensor = TestTensorInt::from([[1, 1, 1], [1, 1, 1]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([true]);
        assert_eq!(data_expected, data_actual);

        // test bool tensor
        let tensor = TestTensorBool::from([[false, true, false], [true, true, true]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([false]);
        assert_eq!(data_expected, data_actual);

        let tensor = TestTensorBool::from([[true, true, true], [true, true, true]]);
        let data_actual = tensor.all().into_data();
        let data_expected = Data::from([true]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_all_dim() {
        let tensor = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
        let data_actual = tensor.all_dim(1).into_data();
        let data_expected = Data::from([[false], [true]]);
        assert_eq!(data_expected, data_actual);

        // test int tensor
        let tensor = TestTensorInt::from([[0, 1, 0], [1, -1, 1]]);
        let data_actual = tensor.all_dim(1).into_data();
        let data_expected = Data::from([[false], [true]]);
        assert_eq!(data_expected, data_actual);

        // test bool tensor
        let tensor = TestTensorBool::from([[false, true, false], [true, true, true]]);
        let data_actual = tensor.all_dim(1).into_data();
        let data_expected = Data::from([[false], [true]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_all_with_bool_from_lower_equal() {
        let tensor1 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]) + 1e-6;
        let tensor2 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]) + 1e-6;

        let ge = tensor1.lower_equal(tensor2);
        let all = ge.clone().all();

        assert_eq!(Data::from([true]), all.clone().into_data());
    }
}
