#[burn_tensor_testgen::testgen(ad_sign)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    /// Example using the sign function with PyTorch:
    // >>> import torch
    // >>> # Create a tensor with requires_grad=True
    // >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    // >>> # Forward pass: Apply the sign function
    // >>> y = torch.sign(x)
    // >>> print("Forward pass:")
    // Forward pass:
    // >>> print("x:", x)
    // x: tensor([-2., -1.,  0.,  1.,  2.], requires_grad=True)
    // >>> print("y:", y)
    // y: tensor([-1., -1.,  0.,  1.,  1.], grad_fn=<SignBackward0>)
    // >>> # Compute the loss (just an example)
    // >>> loss = y.sum()
    // >>> # Backward pass: Compute the gradients
    // >>> loss.backward()
    // >>> print("\nBackward pass:")
    // Backward pass:
    // >>> print("x.grad:", x.grad)
    // x.grad: tensor([0., 0., 0., 0., 0.])

    #[test]
    fn should_diff_sign() {
        let data = Data::<f32, 1>::from([-2.0, -1.0, 0.0, 1.0, 2.0]);

        let device = Default::default();
        let x = TestAutodiffTensor::from_data(data, &device).require_grad();

        let y = x.clone().sign();

        let loss = y.clone().sum();
        let grads = loss.backward();
        let grad = x.grad(&grads).unwrap();

        assert_eq!(y.to_data(), Data::from([-1., -1., 0., 1., 1.]));
        assert_eq!(grad.to_data(), Data::from([0., 0., 0., 0., 0.]));
    }
}
