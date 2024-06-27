#[burn_tensor_testgen::testgen(ad_gelu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, TensorData};

    #[test]
    fn should_diff_gelu() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::<2>::from_floats([[0.0, 1.0], [-3.0, 4.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_floats([[6.0, -0.5], [9.0, 10.0]], &device).require_grad();

        let x = tensor_1.clone().matmul(activation::gelu(tensor_2.clone()));
        let x = tensor_1.clone().matmul(x);
        let grads = x.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.4629, 1.4629], [48.2286, 153.4629]]);
        grad_1.to_data().assert_approx_eq(&expected, 2);

        let expected = TensorData::from([[-15.0000, -1.9895], [17.0000, 17.0000]]);
        grad_2.to_data().assert_approx_eq(&expected, 2);
    }
}
