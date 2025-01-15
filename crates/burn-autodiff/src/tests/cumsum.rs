#[burn_tensor_testgen::testgen(ad_cumsum)]
mod tests {
    use super::*;
    use burn_tensor::{loss, Tensor, TensorData};

    #[test]
    fn should_diff_cumsum() {
        let device = Default::default();
        let tensor_0 =
            TestAutodiffTensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device)
                .require_grad();

        let dim = 1;
        let tensor_1 = tensor_0.clone().cumsum(dim);

        let grads = tensor_1.backward();

        let grad_0 = tensor_0.grad(&grads).unwrap();
        let grad_0_expected = TensorData::from([[3., 2., 1.], [3., 2., 1.]]);
        grad_0.into_data().assert_approx_eq(&grad_0_expected, 2);
    }
}
