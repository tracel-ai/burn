use super::*;
use burn_tensor::TensorData;

#[test]
fn unfold_backward_accumulates_overlaps() {
    let device = Default::default();
    let x = TestAutodiffTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device).require_grad();

    let y = x.clone().unfold::<3, _>(1, 2, 1);
    let loss = y.sum();

    let grads = loss.backward();
    let grad_x = x.grad(&grads).unwrap();

    grad_x
        .to_data()
        .assert_eq(&TensorData::from([[1., 2., 2., 1.]]), false);
}
