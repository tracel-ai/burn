#[burn_tensor_testgen::testgen(dyn_roundtrip)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_dyn_roundtrip_float() {
        let tensor = TestTensor::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

        let roundtrip_tensor = TestTensor::from(TestTensorDyn::from(tensor.clone()));

        assert_eq!(tensor.into_data(), roundtrip_tensor.into_data());
    }

    #[test]
    fn should_support_dyn_roundtrip_int() {
        let tensor = TestTensorInt::from([[0, -1, 2], [3, 4, -5]]);

        let roundtrip_tensor = TestTensorInt::from(TestTensorDyn::from(tensor.clone()));

        assert_eq!(tensor.into_data(), roundtrip_tensor.into_data());
    }

    #[test]
    fn should_support_dyn_roundtrip_bool() {
        let tensor = TestTensorBool::from([[false, false, false], [true, true, false]]);

        let roundtrip_tensor = TestTensorBool::from(TestTensorDyn::from(tensor.clone()));

        assert_eq!(tensor.into_data(), roundtrip_tensor.into_data());
    }
}
