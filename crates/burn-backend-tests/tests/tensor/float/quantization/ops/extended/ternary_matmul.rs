use super::*;
use burn_tensor::quantization::QuantValue;
use burn_tensor::{DType, Element, TensorData, Tolerance};

// BitNet b1.58 ternary weights (`Q2S`, values in {-1, 0, +1}) take the ndarray backend's native
// multiply-free matmul path (`NdArray::q_matmul`): `+1 => add`, `-1 => subtract`, `0 => skip`, then
// one per-tensor scale at the end. These tests pin its guarantee — the result matches the
// dequantize-then-float-matmul path within f32 rounding — and that an all-zero weight is exactly 0.

#[test]
fn q2s_ternary_matmul_matches_dequantized() {
    // Equality is asserted at f32; skip for lower-precision test float types (e.g. f16), where the
    // two summation orders diverge by more than the matmul tolerance.
    if !matches!(FloatElem::dtype(), DType::F32) {
        return;
    }
    let device = Default::default();

    // Float activation [M, K] and a weight [K, N] (constructing a tensor pins the backend's device
    // type, so `device.settings()` below resolves).
    let a = TestTensor::<2>::from_data(
        [
            [0.5, -1.2, 0.3, 2.0],
            [1.0, 0.0, -0.7, 0.4],
            [-2.0, 1.5, 0.1, -0.3],
        ],
        &device,
    );
    let w = TestTensor::<2>::from_data(
        [
            [0.9, -0.2, 0.5],
            [-0.8, 0.1, 0.7],
            [0.0, 0.6, -0.4],
            [0.3, -0.9, 0.2],
        ],
        &device,
    );
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q2S);
    let wq = w.quantize_dynamic(&scheme);

    // Native path (multiply-free skip/add/sub) vs. dequantize-then-float-matmul reference.
    let native = a.clone().matmul(wq.clone());
    let reference = a.matmul(wq.dequantize());

    native
        .into_data()
        .assert_approx_eq::<FloatElem>(&reference.into_data(), Tolerance::relative(2e-2));
}

#[test]
fn q2s_ternary_matmul_all_zero_weight_is_zero() {
    // An all-zero weight quantizes to all-zero ternary, so the native path skips every term and
    // returns exactly 0 — no NaN, no special-casing in the matmul itself.
    let device = Default::default();

    let a = TestTensor::<2>::from_data([[1.0, -2.0, 3.0]], &device);
    let w = TestTensor::<2>::from_data([[0.0], [0.0], [0.0]], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q2S);
    let wq = w.quantize_dynamic(&scheme);

    let out = a.matmul(wq);

    out.into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[0.0f32]]), Tolerance::absolute(1e-6));
}
