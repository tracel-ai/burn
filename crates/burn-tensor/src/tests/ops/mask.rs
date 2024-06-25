#[burn_tensor_testgen::testgen(mask)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_mask_where_ops() {
        let device = Default::default();
        let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );
        let value = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.8, 2.8], [3.8, 4.8]]),
            &device,
        );

        let output = tensor.mask_where(mask, value);
        let expected = TensorData::from([[1.8, 7.0], [2.0, 4.8]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mask_fill_ops() {
        let device = Default::default();
        let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );

        let output = tensor.mask_fill(mask, 2.0);
        let expected = TensorData::from([[2.0, 7.0], [2.0, 2.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_int_mask_where_ops() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 7], [2, 3]], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );
        let value =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::from([[8, 9], [10, 11]]), &device);

        let output = tensor.mask_where(mask, value);
        let expected = TensorData::from([[8, 7], [2, 11]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_int_mask_fill_ops() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 7], [2, 3]], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );

        let output = tensor.mask_fill(mask, 9);
        let expected = TensorData::from([[9, 7], [2, 9]]);

        output.into_data().assert_eq(&expected, false);
    }
}
