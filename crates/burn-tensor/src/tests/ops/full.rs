#[burn_tensor_testgen::testgen(full)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Bool, Int, Shape, Tensor, TensorData};

    #[test]
    fn test_data_full() {
        let tensor = TensorData::full([2, 3], 2.0);

        tensor.assert_eq(&TensorData::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), false);
    }

    #[test]
    fn test_tensor_full() {
        let device = Default::default();
        // Test full with f32
        let tensor = Tensor::<TestBackend, 2>::full([2, 3], 2.1, &device);

        tensor
            .into_data()
            .assert_eq(&TensorData::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]), false);

        // Test full with Int
        let int_tensor = Tensor::<TestBackend, 2, Int>::full([2, 2], 2, &device);

        int_tensor
            .into_data()
            .assert_eq(&TensorData::from([[2, 2], [2, 2]]), false);

        // TODO enable after adding support for bool
        // // Test full with bool
        // let bool_tensor = Tensor::<TestBackend, 2, Bool>::full([2, 2], true, &device);
        // let data_expected = TensorData::from([[true, true], [true, true]]);
        // assert_eq!(data_expected, bool_tensor.into_data());

        // let bool_tensor = Tensor::<TestBackend, 2, Bool>::full([2, 2], false, &device);
        // let data_expected = TensorData::from([[false, false], [false, false]]);
        // assert_eq!(data_expected, bool_tensor.into_data());
    }
}
