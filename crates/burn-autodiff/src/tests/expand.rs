#[burn_tensor_testgen::testgen(ad_expand)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_diff_expand() {
        // Python code to generate the test case values
        // import torch
        // x1 = torch.tensor([4.0, 7.0, 2.0, 3.0], requires_grad=True)
        // x2 = torch.tensor([2.0, 4.5, 7.0, 3.0], requires_grad=True)
        // y = x1.expand(4, 4)
        // z = (x2 * y).sum()
        // z.backward()
        // print("x1", x1.grad)
        // print("x2", x2.grad)

        let device = Default::default();

        let data_1: Data<f32, 1> = Data::from([4.0, 7.0, 2.0, 3.0]);
        let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();

        let data_2: Data<f32, 1> = Data::from([2.0, 4.5, 7.0, 3.0]);
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().expand([4, 4]);

        // Use unsqueeze to make tensor_2 have the same shape as tensor_3
        let tensor_4 = tensor_2.clone().unsqueeze().mul(tensor_3).sum();
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([8., 18., 28., 12.]));
        assert_eq!(grad_2.to_data(), Data::from([16., 28., 8., 12.]));
    }
}
