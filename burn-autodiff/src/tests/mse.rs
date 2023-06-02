#[burn_tensor_testgen::testgen(ad_mse_loss)]
mod tests {
    use super::*;
    use burn_tensor::{loss, Data, Tensor};

    #[test]
    fn test_mse_loss_grad() {
        let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);
        let data_targets = Data::from([[0.8, 0.2], [0.9, 0.1]]);

        let tensor_1 = Tensor::<TestADBackend, 2>::from_data(data_1).require_grad();
        let tensor_2 = Tensor::<TestADBackend, 2>::from_data(data_2).require_grad();
        let tensor_targets = Tensor::<TestADBackend, 2>::from_data(data_targets).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = loss::mse(tensor_3, tensor_targets);

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[58.8999, 85.9], [372.4499, 543.45]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[79.6499, 91.35], [110.2999, 126.7]]), 3);
    }
}
