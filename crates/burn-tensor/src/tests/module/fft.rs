#[burn_tensor_testgen::testgen(module_fft)]
mod tests {
    use super::*;
    use burn_tensor::module::fft;
    use burn_tensor::Shape;

    #[test]
    fn test_fft_1d() {
        let x = TestTensor::from([[[1., 3.], [2.3, -1.]], [[-1., 0.], [-1., 0.]]]);
        println!("x {:?}", x.clone().into_data());
        println!("x_hat {:?}", fft(x.clone()).into_data());
        let x_hat = TestTensor::from([[[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]]]);
        assert_output(fft(x), x_hat);
    }

    fn assert_output(x: TestTensor<3>, y: TestTensor<3>) {
        x.to_data().assert_approx_eq(&y.into_data(), 3);
    }
}
