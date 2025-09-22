#[burn_tensor_testgen::testgen(trunc)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_trunc_ops() {
        let data = TensorData::from([[2.3, -1.7, 0.5], [-0.5, 3.9, -4.2]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.trunc();
        let expected = TensorData::from([[2.0, -1.0, 0.0], [0.0, 3.0, -4.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_truncate_positive_values_like_floor() {
        let data = TensorData::from([1.7, 2.9, 3.1, 4.5]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.trunc();
        let expected = TensorData::from([1.0, 2.0, 3.0, 4.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_truncate_negative_values_like_ceil() {
        let data = TensorData::from([-1.7, -2.9, -3.1, -4.5]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.trunc();
        let expected = TensorData::from([-1.0, -2.0, -3.0, -4.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
