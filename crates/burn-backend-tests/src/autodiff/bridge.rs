use super::*;
use burn_tensor::{DType, Distribution, Tensor};

#[test]
fn test_full_precision() {
    let device = Default::default();
    let x1 = Tensor::<TestAutodiffBackend, 2>::random([32, 32], Distribution::Default, &device)
        .require_grad();
    let x2 = Tensor::<TestAutodiffBackend, 2>::random([32, 32], Distribution::Default, &device)
        .require_grad();
    let dtype = x1.dtype();

    let x3 = x1.clone().cast(DType::F32);
    let x4 = x2.clone().cast(DType::F32);

    let x5 = x3.matmul(x4);
    let x6 = x5.cast(dtype);
    let x7 = x6 * x1.clone() / x2.clone();

    let mut grads = x7.backward();

    let x1_grad = x1.grad(&grads);
    let x2_grad = x2.grad(&grads);

    assert!(x1_grad.is_some());
    assert!(x2_grad.is_some());
}
