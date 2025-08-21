#[burn_tensor_testgen::testgen(unit)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Shape, Tensor, TensorData, backend::Backend};

    #[test]
    fn test_tensor_unit() {
        let device = Default::default();
        // Test full with f32
        let tensor = TestTensor::<2>::unit(2.1, &device);
        tensor
            .into_data()
            .assert_eq(&TensorData::from([[2.1]]), false);

        // Test full with Int
        let int_tensor = TestTensorInt::<3>::unit(2, &device);
        int_tensor
            .into_data()
            .assert_eq(&TensorData::from([[[2]]]), false);

        // Test full with bool
        let bool_tensor = TestTensorBool::<1>::unit(true, &device);
        bool_tensor
            .into_data()
            .assert_eq(&TensorData::from([true]), false);
    }
}
