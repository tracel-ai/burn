#[burn_tensor_testgen::testgen(nan)]
mod tests {
    use super::*;
    use burn_tensor::{cast::ToElement, Int, Tensor, TensorData};

    #[test]
    #[ignore = "https://github.com/tracel-ai/burn/issues/2089"]
    fn is_nan() {
        let no_nan = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let no_nan_expected =
            TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);

        let with_nan = TestTensor::<2>::from([[0.0, f32::NAN, 2.0], [f32::NAN, 4.0, 5.0]]);
        let with_nan_expected =
            TestTensorBool::<2>::from([[false, true, false], [true, false, false]]);

        assert_eq!(no_nan_expected.into_data(), no_nan.is_nan().into_data());

        assert_eq!(with_nan_expected.into_data(), with_nan.is_nan().into_data());
    }

    #[test]
    #[ignore = "https://github.com/tracel-ai/burn/issues/2089"]
    fn contains_nan() {
        let no_nan = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        assert!(!no_nan.contains_nan().into_scalar().to_bool());

        let with_nan = TestTensor::<2>::from([[0.0, f32::NAN, 2.0], [3.0, 4.0, 5.0]]);
        assert!(with_nan.contains_nan().into_scalar().to_bool());
    }
}
