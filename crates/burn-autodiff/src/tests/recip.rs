#[burn_tensor_testgen::testgen(ad_recip)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, TensorData};

    #[test]
    fn should_diff_recip() {
        let data = TensorData::from([2.0, 5.0, 0.4]);

        let tensor = TestAutodiffTensor::<1>::from_data(data, &Default::default()).require_grad();
        let tensor_out = tensor.clone().recip();

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        let expected = TensorData::from([0.5, 0.2, 2.5])
            .convert::<<TestAutodiffBackend as Backend>::FloatElem>();
        tensor_out.into_data().assert_eq(&expected, true);

        let expected = TensorData::from([-0.25, -0.04, -6.25])
            .convert::<<TestAutodiffBackend as Backend>::FloatElem>();
        grad.to_data().assert_approx_eq(&expected, 3);
    }
}
