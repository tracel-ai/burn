use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::prepare_linear_weight;

// `[64, 128]` and `[2, 32, 64]` are divided by any tile a backend stores a weight in; `[3, 5]` by
// none, so it stays as it lies.

#[test]
fn prepare_linear_weight_keeps_the_values() {
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
        prepare_linear_weight(weight.clone()).into_data(),
        weight.into_data()
    );
    assert_eq!(
        prepare_linear_weight(batched.clone()).into_data(),
        batched.into_data()
    );
    assert_eq!(
        prepare_linear_weight(odd.clone()).into_data(),
        odd.into_data()
    );
}

#[test]
fn prepare_linear_weight_gives_the_same_product() {
    let device = Default::default();
    let x = TestTensorInt::<1>::arange(0..16 * 64, &device)
        .float()
        .reshape([16, 64])
        .cos();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin();

    let laid = x.clone().matmul(prepare_linear_weight(weight.clone()));

    laid.into_data().assert_approx_eq::<FloatElem>(
        &x.matmul(weight).into_data(),
        Tolerance::rel_abs(1e-2, 1e-3),
    );
}

#[test]
fn prepare_linear_weight_takes_elementwise_ops() {
    let device = Default::default();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin();

    let stepped = (prepare_linear_weight(weight.clone()) * 2.0 + 1.0).exp();

    stepped.into_data().assert_approx_eq::<FloatElem>(
        &(weight * 2.0 + 1.0).exp().into_data(),
        Tolerance::default(),
    );
}
