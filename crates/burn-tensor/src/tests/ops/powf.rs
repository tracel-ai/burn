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

    fn outer<B: burn_tensor::backend::Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 2> {
        a.unsqueeze_dim::<2>(1) * b.unsqueeze_dim::<2>(0)
    }

    #[test]
    fn should_support_powf_scalar_tensor() {
        let device = Default::default();
        let head_dim = 64;
        let seq_len = 1024;
        let base = 10000;

        let channel_range: Tensor<TestBackend, 1> =
            Tensor::arange_step(0..head_dim as i64, 2, &device).float();
        let base: Tensor<TestBackend, 1> = Tensor::from_data([base as f32], &device);
        let inv_freq: Tensor<TestBackend, 1> = base.powf(-channel_range / head_dim as f32);

        let t: Tensor<TestBackend, 1> = Tensor::arange(0..seq_len as i64, &device).float();

        let freqs = outer(t, inv_freq);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();
    }
}
