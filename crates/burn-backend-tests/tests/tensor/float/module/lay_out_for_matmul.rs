use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::lay_out_for_matmul;

// `[64, 128]` and `[2, 32, 64]` are divided by any tile a backend stores a weight in; `[3, 5]` by
// none, so it stays as it lies.

#[test]
fn lay_out_for_matmul_keeps_the_values() {
    let device = Default::default();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin();
    let batched = TestTensorInt::<1>::arange(0..2 * 32 * 64, &device)
        .float()
        .reshape([2, 32, 64])
        .sin();
    let odd = TestTensorInt::<1>::arange(0..15, &device)
        .float()
        .reshape([3, 5]);

    assert_eq!(
        lay_out_for_matmul(weight.clone()).into_data(),
        weight.into_data()
    );
    assert_eq!(
        lay_out_for_matmul(batched.clone()).into_data(),
        batched.into_data()
    );
    assert_eq!(lay_out_for_matmul(odd.clone()).into_data(), odd.into_data());
}

#[test]
fn lay_out_for_matmul_gives_the_same_product() {
    let device = Default::default();
    let x = TestTensorInt::<1>::arange(0..16 * 64, &device)
        .float()
        .reshape([16, 64])
        .cos();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin();

    let laid = x.clone().matmul(lay_out_for_matmul(weight.clone()));

    laid.into_data().assert_approx_eq::<FloatElem>(
        &x.matmul(weight).into_data(),
        Tolerance::rel_abs(1e-2, 1e-3),
    );
}

#[test]
fn lay_out_for_matmul_takes_elementwise_ops() {
    let device = Default::default();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin();

    let stepped = (lay_out_for_matmul(weight.clone()) * 2.0 + 1.0).exp();

    stepped.into_data().assert_approx_eq::<FloatElem>(
        &(weight * 2.0 + 1.0).exp().into_data(),
        Tolerance::default(),
    );
}
