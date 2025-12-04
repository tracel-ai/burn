use crate::*;
use burn_tensor::{Tensor, TensorData, Tolerance, loss};

#[test]
fn test_cross_entropy_loss_grad() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);
    let data_targets = TensorData::from([[0.8, 0.2], [0.9, 0.1]]);

    let device = Default::default();
    let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(data_1, &device).require_grad();
    let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(data_2, &device).require_grad();
    let tensor_targets =
        Tensor::<TestAutodiffBackend, 2>::from_data(data_targets, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = loss::cross_entropy_with_logits(tensor_3, tensor_targets);

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::permissive();
    let expected = TensorData::from([[0.26553, 0.26553], [0.44954, 0.44954]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[-1.34863, 1.34863], [-2.06371, 2.06371]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
