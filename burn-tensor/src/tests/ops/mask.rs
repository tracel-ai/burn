#[burn_tensor_testgen::testgen(mask)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Tensor};

    #[test]
    fn should_support_mask_where_ops() {
        let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]]);
        let mask =
            Tensor::<TestBackend, 2, Bool>::from_bool(Data::from([[true, false], [false, true]]));
        let value = Tensor::<TestBackend, 2>::from_data(Data::from([[8.8, 8.8], [8.8, 8.8]]));

        let data_actual = tensor.mask_where(mask, value).into_data();

        let data_expected = Data::from([[8.8, 7.0], [2.0, 8.8]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mask_fill_ops() {
        let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]]);
        let mask =
            Tensor::<TestBackend, 2, Bool>::from_bool(Data::from([[true, false], [false, true]]));

        let data_actual = tensor.mask_fill(mask, 2.0).to_data();

        let data_expected = Data::from([[2.0, 7.0], [2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }
}
