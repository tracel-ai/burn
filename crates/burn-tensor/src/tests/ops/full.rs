#[burn_tensor_testgen::testgen(full)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Shape, Tensor, TensorData, backend::Backend};

    #[test]
    fn test_data_full() {
        let tensor = TensorData::full([2, 3], 2.0);

        tensor.assert_eq(&TensorData::from([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), false);
    }

    #[test]
    fn test_tensor_full() {
        let device = Default::default();
        // Test full with f32
        let tensor = TestTensor::<2>::full([2, 3], 2.1, &device);
        tensor
            .into_data()
            .assert_eq(&TensorData::from([[2.1, 2.1, 2.1], [2.1, 2.1, 2.1]]), false);

        // Test full with Int
        let int_tensor = TestTensorInt::<2>::full([2, 2], 2, &device);
        int_tensor
            .into_data()
            .assert_eq(&TensorData::from([[2, 2], [2, 2]]), false);

        // Test full with bool
        let bool_tensor = TestTensorBool::<2>::full([2, 2], true, &device);
        bool_tensor
            .into_data()
            .assert_eq(&TensorData::from([[true, true], [true, true]]), false);

        let bool_tensor = TestTensorBool::<2>::full([2, 2], false, &device);
        bool_tensor
            .into_data()
            .assert_eq(&TensorData::from([[false, false], [false, false]]), false);
    }
}
