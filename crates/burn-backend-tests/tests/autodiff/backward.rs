use super::*;
use burn_tensor::{TensorData, module::embedding};

#[test]
fn test_embedding_backward() {
    let weights = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let indices = TensorData::from([[0, 1], [1, 1]]);
    let x = TensorData::from([
        [[1.0, 2.0], [4.0, 5.0], [3.0, 4.0]],
        [[4.0, 5.0], [8.0, 5.0], [1.0, 9.0]],
    ]);
    let device = AutodiffDevice::new();
    let weights = TestTensor::<2>::from_data(weights, &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(indices, &device);
    let x = TestTensor::<3>::from_data(x, &device).require_grad();

    let output = embedding(weights.clone(), indices);
    let output = output.matmul(x);
    let grads = output.backward();

    let grad = weights.grad(&grads).unwrap();
    grad.to_data()
        .assert_eq(&TensorData::from([[3., 9., 7.], [21., 35., 27.]]), false);
}
