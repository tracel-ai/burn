#[burn_tensor_testgen::testgen(ad_cast)]
mod tests {
    use super::*;
    use burn_tensor::{DType, Tensor, TensorData};

    #[test]
    fn cast_keeps_gradient_flow() {
        let device = Default::default();
        let x = Tensor::<TestAutodiffBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        )
        .require_grad();

        let dtype = x.dtype();
        let y = x.clone().cast(dtype);
        let z = y.sum();

        let grads = z.backward();
        let grad_x = x.grad(&grads).unwrap();

        grad_x
            .to_data()
            .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
    }
}
