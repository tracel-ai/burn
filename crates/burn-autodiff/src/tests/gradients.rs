#[burn_tensor_testgen::testgen(gradients)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Distribution};

    #[test]
    fn should_update_tensor_when_grad_replace() {
        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::random([32, 32], Distribution::Default, &device)
            .require_grad();
        let tensor_2 = TestAutodiffTensor::random([32, 32], Distribution::Default, &device);

        let x = tensor_1.clone().matmul(activation::gelu(tensor_2));
        let mut grads = x.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();

        let grad_1_updated =
            TestAutodiffTensor::random([32, 32], Distribution::Default, &device).require_grad();
        tensor_1.grad_replace(&mut grads, grad_1_updated.clone().inner());

        let grad_1_new = tensor_1.grad(&grads).unwrap();

        assert_ne!(grad_1_new.to_data(), grad_1.into_data());
        assert_eq!(grad_1_new.into_data(), grad_1_updated.into_data());
    }
}
