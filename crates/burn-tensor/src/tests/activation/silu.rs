#[burn_tensor_testgen::testgen(silu)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, activation};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_silu() {
        let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

        let output = activation::silu(tensor);
        let expected = TensorData::from([[0.73106, 1.76159], [2.85772, 3.92806]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
