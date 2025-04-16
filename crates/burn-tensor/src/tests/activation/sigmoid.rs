#[burn_tensor_testgen::testgen(sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation, tests::Numeric};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_sigmoid() {
        let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

        let output = activation::sigmoid(tensor);
        let expected = TensorData::from([[0.731059, 0.999089], [0.999998, 0.047426]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_sigmoid_overflow() {
        let tensor = TestTensor::<1>::from([FloatType::MAX, FloatType::MIN]);

        let output = activation::sigmoid(tensor);
        let expected = TensorData::from([1.0, 0.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
