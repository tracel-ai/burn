#[burn_tensor_testgen::testgen(ad_cat)]
mod tests {
    use super::*;

    #[test]
    fn should_diff_cat() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::<2>::from_data([[2.0, -1.0], [5.0, 2.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_data([[5.0, 4.0], [-1.0, 4.0]], &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let mut tensor_1_list = Vec::new();
        let mut tensor_2_list = Vec::new();

        for i in 0..2 {
            tensor_1_list.push(tensor_1.clone().slice([i..i + 1]));
            tensor_2_list.push(tensor_2.clone().slice([i..i + 1]));
        }

        let tensor_1_cat = TestAutodiffTensor::cat(tensor_1_list.clone(), 0);
        let tensor_2_cat = TestAutodiffTensor::cat(tensor_2_list.clone(), 0);

        let tensor_3_cat = tensor_1_cat.clone().matmul(tensor_2_cat.clone());
        let grads = tensor_3_cat.backward();

        let grad_1_slice_1 = tensor_1.grad(&grads).unwrap().slice([0..1]);
        let grad_1_slice_2 = tensor_1.grad(&grads).unwrap().slice([1..2]);

        let grad_2_slice_1 = tensor_2.grad(&grads).unwrap().slice([0..1]);
        let grad_2_slice_2 = tensor_2.grad(&grads).unwrap().slice([1..2]);

        grad_1
            .clone()
            .slice([0..1])
            .to_data()
            .assert_approx_eq(&grad_1_slice_1.to_data(), 3);
        grad_1
            .slice([1..2])
            .to_data()
            .assert_approx_eq(&grad_1_slice_2.to_data(), 3);

        grad_2
            .clone()
            .slice([0..1])
            .to_data()
            .assert_approx_eq(&grad_2_slice_1.to_data(), 3);
        grad_2
            .slice([1..2])
            .to_data()
            .assert_approx_eq(&grad_2_slice_2.to_data(), 3);
    }

    #[test]
    fn should_diff_cat_more_than_1_dim() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_data([[2.0, -1.0], [5.0, 2.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_data([[5.0, 4.0], [-1.0, 4.0], [4.0, 1.0]], &device)
                .require_grad();

        // Concat a tensor [2, 2] with another tensor [3, 2] along dim 0.
        // The resulting tensor should be [5, 2]
        let tensor_3 = TestAutodiffTensor::cat(vec![tensor_1.clone(), tensor_2.clone()], 0);
        assert_eq!(tensor_3.dims(), [5, 2]);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(tensor_1.dims(), grad_1.dims());
        assert_eq!(tensor_2.dims(), grad_2.dims());
    }
}
