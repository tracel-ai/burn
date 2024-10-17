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
    fn should_support_mask_where_broadcast_int() {
        let device = Default::default();
        // When broadcasted, the input [[2, 3], [4, 5]] is repeated 4 times
        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..6, &device).reshape([1, 2, 2]);
        let mask = Tensor::<TestBackend, 3, Bool>::from_bool(
            TensorData::from([
                [[true, false], [false, true]],
                [[false, true], [true, false]],
                [[false, false], [false, false]],
                [[true, true], [true, true]],
            ]),
            &device,
        );
        let value = Tensor::<TestBackend, 3, Int>::ones([4, 2, 2], &device);

        let output = tensor.mask_where(mask, value);
        let expected = TensorData::from([
            [[1, 3], [4, 1]],
            [[2, 1], [1, 5]],
            [[2, 3], [4, 5]],
            [[1, 1], [1, 1]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_mask_where_broadcast() {
        let device = Default::default();
        // When broadcasted, the input [[2, 3], [4, 5]] is repeated 4 times
        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..6, &device).reshape([1, 2, 2]);
        let mask = Tensor::<TestBackend, 3, Bool>::from_bool(
            TensorData::from([
                [[true, false], [false, true]],
                [[false, true], [true, false]],
                [[false, false], [false, false]],
                [[true, true], [true, true]],
            ]),
            &device,
        );
        let value = Tensor::<TestBackend, 3>::ones([4, 2, 2], &device);

        let output = tensor.float().mask_where(mask, value);
        let expected = TensorData::from([
            [[1., 3.], [4., 1.]],
            [[2., 1.], [1., 5.]],
            [[2., 3.], [4., 5.]],
            [[1., 1.], [1., 1.]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_handle_mask_where_nans() {
        let device = Default::default();
        let tensor = TestTensor::from_data(
            [
                [f32::NAN, f32::NAN, f32::NAN],
                [f32::NAN, f32::NAN, f32::NAN],
                [f32::NAN, f32::NAN, f32::NAN],
            ],
            &device,
        );
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([
                [true, true, true],
                [true, true, false],
                [false, false, false],
            ]),
            &device,
        );
        let value = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]),
            &device,
        );

        let output = tensor.mask_where(mask, value);
        let expected = TensorData::from([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, f32::NAN],
            [f32::NAN, f32::NAN, f32::NAN],
        ]);

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

    #[test]
    fn float_mask_fill_infinite() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data(
            [
                [f32::NEG_INFINITY, f32::NEG_INFINITY],
                [f32::NEG_INFINITY, f32::NEG_INFINITY],
            ],
            &device,
        );
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );

        let output = tensor.mask_fill(mask, 10.0f32);
        let expected = TensorData::from([[10f32, f32::NEG_INFINITY], [f32::NEG_INFINITY, 10f32]]);

        output.into_data().assert_eq(&expected, false);
    }
}
