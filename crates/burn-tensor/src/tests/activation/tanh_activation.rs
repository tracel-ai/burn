#[burn_tensor_testgen::testgen(tanh_activation)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_tanh() {
        let tensor = TestTensor::<2>::from([[1., 2.], [3., 4.]]);

        let output = activation::tanh(tensor);
        let expected = TensorData::from([[0.761594, 0.964028], [0.995055, 0.999329]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
