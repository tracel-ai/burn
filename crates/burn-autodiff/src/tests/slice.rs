#[burn_tensor_testgen::testgen(ad_slice)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_diff_matmul_with_slice() {
        let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = TensorData::from([[4.0, 7.0, 100.0], [2.0, 3.0, 15.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_2.clone().slice([0..2, 0..2]);
        let tensor_4 = tensor_1.clone().matmul(tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_eq(&TensorData::from([[11.0, 5.0], [11.0, 5.0]]), false);
        grad_2.to_data().assert_eq(
            &TensorData::from([[3.0, 3.0, 0.0], [10.0, 10.0, 0.0]]),
            false,
        );
    }

    #[test]
    fn should_diff_matmul_with_slice_assign() {
        let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_assigned = TensorData::from([[9.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
        let tensor_assigned = TestAutodiffTensor::from_data(data_assigned, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.slice_assign([0..1, 0..1], tensor_assigned);
        let tensor_5 = tensor_4.matmul(tensor_1.clone());

        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_eq(&TensorData::from([[58.0, 38.0], [118.0, 82.0]]), false);
        grad_2
            .to_data()
            .assert_eq(&TensorData::from([[16.0, 15.0], [24.0, 50.0]]), false);
    }

    #[test]
    fn should_diff_matmul_with_slice_assign_complex() {
        let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3 = TensorData::from([[9.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();

        let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_5 = tensor_2.clone().slice([0..1, 0..1]);
        let tensor_6 = tensor_5.mul(tensor_3.clone());
        let tensor_7 = tensor_4.slice_assign([0..1, 0..1], tensor_6);
        let tensor_8 = tensor_7.matmul(tensor_1.clone());

        let grads = tensor_8.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();
        let grad_3 = tensor_3.grad(&grads).unwrap();

        grad_3
            .to_data()
            .assert_eq(&TensorData::from([[32.0]]), false);
        grad_1
            .to_data()
            .assert_eq(&TensorData::from([[85.0, 65.0], [118.0, 82.0]]), false);
        grad_2
            .to_data()
            .assert_eq(&TensorData::from([[88.0, 15.0], [24.0, 50.0]]), false);
    }

    #[test]
    fn slice_assign_diff_should_give_same_results_as_cat() {
        let data_1 = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[5.0, 6.0], [7.0, 8.0]]);
        let data_3 = TensorData::from([[14.0, 97.0, 100.0, 9.0], [2.0, 3.0, 15.0, 7.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
        let tensor_3 = TestAutodiffTensor::from_data(data_3, &device);

        let slice_assign_output = TestAutodiffTensor::zeros([2, 4], &Default::default());
        let slice_assign_output = slice_assign_output.slice_assign([0..2, 0..2], tensor_1.clone());
        let slice_assign_output = slice_assign_output.slice_assign([0..2, 2..4], tensor_2.clone());
        let slice_assign_output = slice_assign_output / tensor_3.clone();

        let cat_output = TestAutodiffTensor::cat(vec![tensor_1.clone(), tensor_2.clone()], 1);
        let cat_output = cat_output / tensor_3;

        slice_assign_output
            .to_data()
            .assert_approx_eq(&cat_output.to_data(), 3);

        let slice_assign_grads = slice_assign_output.backward();
        let cat_grads = cat_output.backward();

        let slice_assign_grad_1 = tensor_1.grad(&slice_assign_grads).unwrap();
        let slice_assign_grad_2 = tensor_2.grad(&slice_assign_grads).unwrap();
        let cat_grad_1 = tensor_1.grad(&cat_grads).unwrap();
        let cat_grad_2 = tensor_2.grad(&cat_grads).unwrap();

        slice_assign_grad_1
            .to_data()
            .assert_approx_eq(&cat_grad_1.to_data(), 3);
        slice_assign_grad_2
            .to_data()
            .assert_approx_eq(&cat_grad_2.to_data(), 3);
    }
}
