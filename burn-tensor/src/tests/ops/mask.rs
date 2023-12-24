#[burn_tensor_testgen::testgen(mask)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_mask_where_ops() {
        let tensor = TestTensor::from_data_devauto([[1.0, 7.0], [2.0, 3.0]]);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool_devauto(Data::from([
            [true, false],
            [false, true],
        ]));
        let value =
            Tensor::<TestBackend, 2>::from_data_devauto(Data::from([[1.8, 2.8], [3.8, 4.8]]));

        let data_actual = tensor.mask_where(mask, value).into_data();

        let data_expected = Data::from([[1.8, 7.0], [2.0, 4.8]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mask_fill_ops() {
        let tensor = TestTensor::from_data_devauto([[1.0, 7.0], [2.0, 3.0]]);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool_devauto(Data::from([
            [true, false],
            [false, true],
        ]));

        let data_actual = tensor.mask_fill(mask, 2.0).to_data();

        let data_expected = Data::from([[2.0, 7.0], [2.0, 2.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_int_mask_where_ops() {
        let tensor = Tensor::<TestBackend, 2, Int>::from_data_devauto([[1, 7], [2, 3]]);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool_devauto(Data::from([
            [true, false],
            [false, true],
        ]));
        let value =
            Tensor::<TestBackend, 2, Int>::from_data_devauto(Data::from([[8, 9], [10, 11]]));

        let data_actual = tensor.mask_where(mask, value).into_data();

        let data_expected = Data::from([[8, 7], [2, 11]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_int_mask_fill_ops() {
        let tensor = Tensor::<TestBackend, 2, Int>::from_data_devauto([[1, 7], [2, 3]]);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool_devauto(Data::from([
            [true, false],
            [false, true],
        ]));

        let data_actual = tensor.mask_fill(mask, 9).to_data();

        let data_expected = Data::from([[9, 7], [2, 9]]);
        assert_eq!(data_expected, data_actual);
    }
}
