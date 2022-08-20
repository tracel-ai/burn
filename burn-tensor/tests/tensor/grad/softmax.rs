use super::super::TestADBackend;
use burn_tensor::{af, Data, Tensor};

#[test]
fn test_softmax_grad() {
    let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);
    let tensor_1 = Tensor::<TestADBackend, 2>::from_data(data_1);
    let tensor_2 = Tensor::<TestADBackend, 2>::from_data(data_2);

    let tensor_3 = tensor_1.matmul(&tensor_2);
    let tensor_4 = af::softmax(&tensor_3, 1).matmul(&tensor_2);

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_approx_eq(&Data::from([[1.1797, 1.1797], [0.0055, 0.0055]]), 3);
    grad_2
        .to_data()
        .assert_approx_eq(&Data::from([[0.2534, 0.2862], [0.5286, 2.9317]]), 3);
}

#[test]
fn test_log_softmax_grad() {
    let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);
    let tensor_1 = Tensor::<TestADBackend, 2>::from_data(data_1);
    let tensor_2 = Tensor::<TestADBackend, 2>::from_data(data_2);

    let tensor_3 = tensor_1.matmul(&tensor_2);
    let tensor_4 = af::log_softmax(&tensor_3, 1).matmul(&tensor_2);

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_approx_eq(&Data::from([[-4.3939, -4.3939], [-12.9709, -12.9709]]), 3);
    grad_2
        .to_data()
        .assert_approx_eq(&Data::from([[30.5984, -47.2267], [55.9631, -56.5914]]), 3);
}
