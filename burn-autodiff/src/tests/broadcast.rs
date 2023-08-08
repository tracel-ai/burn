#[burn_tensor_testgen::testgen(ad_broadcast)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Distribution, Int, Shape, Tensor};

    #[test]
    fn should_handle_broadcast_during_backward() {
        let x: Tensor<TestADBackend, 2> = Tensor::from_data(
            Tensor::<TestADBackend, 1, Int>::arange(0..6)
                .into_data()
                .convert(),
        )
        .reshape([2, 3])
        .require_grad();
        let label: Tensor<TestADBackend, 2> = Tensor::from_data(
            Tensor::<TestADBackend, 1, Int>::arange(0..2)
                .into_data()
                .convert(),
        )
        .reshape([2, 1])
        .require_grad();

        let weights: Tensor<TestADBackend, 2> = Tensor::from_data(
            Tensor::<TestADBackend, 1, Int>::arange(0..3)
                .into_data()
                .convert(),
        )
        .reshape([3, 1])
        .require_grad();

        let bias: Tensor<TestADBackend, 2> = Tensor::from_data(
            Tensor::<TestADBackend, 1, Int>::arange(0..3)
                .into_data()
                .convert(),
        )
        .reshape([1, 3])
        .require_grad();

        let y = x.clone().matmul(weights.clone()).add(bias.clone());
        let loss = y.clone().sub(label).powf(2.0).sum();

        let grads = loss.backward();

        let weights_grad = weights.grad(&grads).unwrap();
        let bias_grad = bias.grad(&grads).unwrap();
        let x_grad = x.grad(&grads).unwrap();

        weights_grad
            .into_data()
            .assert_approx_eq(&Data::new(vec![252., 372., 492.], Shape::new([3, 1])), 3);
        bias_grad
            .into_data()
            .assert_approx_eq(&Data::from([[36., 40., 44.]]), 3);
        x_grad
            .into_data()
            .assert_approx_eq(&Data::from([[0., 36., 72.], [0., 84., 168.]]), 3);
    }

    #[test]
    fn grad_same_shape_as_forward_tensor() {
        let x: Tensor<TestADBackend, 2> =
            Tensor::random([2, 1], Distribution::Default).require_grad();
        let y: Tensor<TestADBackend, 2> =
            Tensor::random([2, 3], Distribution::Default).require_grad();
        let z = x.clone().add(y);

        let grads = z.backward();
        let x_grad = x.grad(&grads).unwrap();

        assert!(x_grad.shape() == x.shape());
    }
}
