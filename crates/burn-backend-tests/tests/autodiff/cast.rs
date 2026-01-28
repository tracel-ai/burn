use super::*;
use burn_tensor::{DType, Tensor, TensorData};

#[cfg(feature = "std")]
use burn_backend_tests::might_panic;

// Skip on metal - F64 not supported
#[cfg(all(feature = "std", not(feature = "metal")))]
#[might_panic(reason = "Unsupported precision for fusion")]
#[test]
fn cast_keeps_gradient_flow() {
    let device = Default::default();
    let x = Tensor::<TestAutodiffBackend, 2>::from_data(
        TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
        &device,
    )
    .require_grad();

    let y = x.clone().cast(DType::F64);
    let z = y.sum();

    let grads = z.backward();
    let grad_x = x.grad(&grads).unwrap();

    grad_x
        .to_data()
        .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
}
