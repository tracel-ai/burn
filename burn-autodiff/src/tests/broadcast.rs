#[burn_tensor_testgen::testgen(ad_broadcast)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn should_handle_broacast_during_backward() {
        let x: Tensor<TestADBackend, 2> =
            Tensor::random([2, 3], Distribution::Standard).require_grad();
        let label: Tensor<TestADBackend, 2> =
            Tensor::random([2, 1], Distribution::Standard).require_grad();

        let weights: Tensor<TestADBackend, 2> =
            Tensor::random([3, 1], Distribution::Standard).require_grad();
        let bias: Tensor<TestADBackend, 2> =
            Tensor::random([1, 3], Distribution::Standard).require_grad();

        let y = x.matmul(weights.clone()).add(bias.clone());
        let loss = y.clone().sub(label).powf(2.0).sum();

        let _gradients = loss.backward(); // Should not panic
    }

    #[test]
    fn grad_should_be_same_size_as_tensor() {
        let x: Tensor<TestADBackend, 2> =
            Tensor::random([2, 1], Distribution::Standard).require_grad();
        let y: Tensor<TestADBackend, 2> =
            Tensor::random([1, 3], Distribution::Standard).require_grad();
        let z = x.clone().add(y);

        let grads = z.backward();
        let x_grad = x.grad(&grads).unwrap();

        assert!(x_grad.shape() == x.shape());
    }
}
