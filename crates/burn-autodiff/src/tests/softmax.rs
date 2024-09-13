#[burn_tensor_testgen::testgen(ad_softmax)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.1797, 1.1797], [0.0055, 0.0055]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[0.2534, 0.2862], [0.5286, 2.9317]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_log_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::log_softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[-4.3939, -4.3939], [-12.9709, -12.9709]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[30.5984, -47.2267], [55.9631, -56.5914]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_quiet_softmax_grad() {
        let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

        let device = Default::default();
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::softmax(tensor_3, 1).matmul(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.1797, 1.1797], [0.0055, 0.0055]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[0.2534, 0.2862], [0.5286, 2.9317]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);
    }
}
