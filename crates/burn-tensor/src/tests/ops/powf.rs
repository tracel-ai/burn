#[burn_tensor_testgen::testgen(powf)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_powf_ops() {
        let data = TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let pow = TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 2.0]]);
        let tensor_pow = TestTensor::<2>::from_data(pow, &Default::default());

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1.0, 1.0, 4.0], [27.0, 256.0, 25.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_power() {
        let data = TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let pow = TensorData::from([[-0.95, -0.67, -0.45], [-0.24, -0.5, -0.6]]);
        let tensor_pow = TestTensor::<2>::from_data(pow, &Default::default());

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1., 1., 0.73204285], [0.76822936, 0.5, 0.38073079]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_values_with_even_power() {
        let data = TensorData::from([[1.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let pow = TensorData::from([[2.0, 2.0, 4.0], [4.0, 4.0, 2.0]]);
        let tensor_pow = TestTensor::<2>::from_data(pow, &Default::default());

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1.0, 1.0, 16.0], [81.0, 256.0, 25.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_values_with_odd_power() {
        let data = TensorData::from([[1.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let pow = TensorData::from([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);
        let tensor_pow = TestTensor::<2>::from_data(pow, &Default::default());

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1.0, -1.0, -8.0], [-27.0, -64.0, -125.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_powf_broadcasted() {
        let device = Default::default();
        let tensor_1 = TestTensor::<1>::from_floats([2.0, 3.0, 4.0], &device);
        let tensor_2 = Tensor::from_floats([1.0], &device);

        // Broadcast rhs
        let output = tensor_1.clone().powf(tensor_2.clone());
        output
            .into_data()
            .assert_approx_eq::<FT>(&tensor_1.to_data(), Tolerance::default());

        // Broadcast lhs
        let output = tensor_2.powf(tensor_1);
        output
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([1.0, 1.0, 1.0]), Tolerance::default());
    }
}
