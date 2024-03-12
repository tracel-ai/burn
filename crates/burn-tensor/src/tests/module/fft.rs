#[burn_tensor_testgen::testgen(module_fft)]
mod tests {
    use super::*;
    use burn_tensor::module::fft;
    use burn_tensor::Shape;

    #[test]
    fn test_fft_1d() {
        assert_output(
            TestTensor::from([[[1., 0.], [0., 0.]], [[-1., 0.], [-1., 0.]]]),
            TestTensor::from([[[1., 0.], [0., 0.]], [[-1., 0.], [-1., 0.]]]),
            // TestTensor::from([[[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]]]),
        )
    }

    fn assert_output(x: TestTensor<3>, y: TestTensor<3>) {
        x.to_data().assert_approx_eq(&y.into_data(), 3);
    }
}
