use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::prepare_linear_weight;

#[test]
fn should_diff_through_prepare_linear_weight() {
    let device = AutodiffDevice::new();
    let x = TestTensorInt::<1>::arange(0..16 * 64, &device)
        .float()
        .reshape([16, 64])
        .cos();
    let weight = TestTensorInt::<1>::arange(0..64 * 128, &device)
        .float()
        .reshape([64, 128])
        .sin()
        .require_grad();

    let laid = x.clone().matmul(prepare_linear_weight(weight.clone()));
    let grads = laid.sum().backward();
    let plain = weight.clone().detach().require_grad();
    let plain_grads = x.matmul(plain.clone()).sum().backward();

    weight
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &plain.grad(&plain_grads).unwrap().into_data(),
            Tolerance::rel_abs(1e-2, 1e-3),
        );
}
