use super::*;
use burn_tensor::{DType, TensorData, activation};

// Half-precision activations are evaluated in f32 and rounded once, as in PyTorch. Each case
// below fails in the half type itself: an intermediate overflows (`x³` in the tanh form of GELU,
// `x · (1 + erf)` at 49152 in f16) or a tail cancels (`1 + tanh(u)` and `1 + erf(u)` for very
// negative `x`, `log1p` of a tiny `exp(x)`). The expected values are exact to f64, and the inputs
// are exact in f16 and bf16.

/// Checks `activation` in f16 and bf16 against `expected`, within 1% of each value.
fn assert_half_matches(
    activation: impl Fn(TestTensor<1>) -> TestTensor<1>,
    inputs: &[f32],
    expected: &[f64],
) {
    let device = burn_tensor::Device::default();
    for dtype in [DType::F16, DType::BF16] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        let input = TestTensor::<1>::from_data(
            TensorData::new(inputs.to_vec(), [inputs.len()]),
            (&device, dtype),
        );
        let output = activation(input).into_data().convert::<f32>();
        for ((&x, &got), &want) in inputs
            .iter()
            .zip(output.as_slice::<f32>().unwrap())
            .zip(expected)
        {
            assert!(
                (got as f64 - want).abs() <= 1e-2 * want.abs(),
                "{dtype:?} at x = {x}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn should_compute_gelu_in_f32_for_half() {
    assert_half_matches(
        activation::gelu,
        &[49152.0, -4.0, -3.0, 1.0],
        &[
            49152.0,
            -1.2668496733247986e-4,
            -4.049694094890288e-3,
            0.8413447460685428,
        ],
    );
}

#[test]
fn should_compute_gelu_approximate_in_f32_for_half() {
    assert_half_matches(
        activation::gelu_approximate,
        &[49152.0, -4.0, -3.0, -2.0],
        &[
            49152.0,
            -7.024594819237266e-5,
            -3.637392081773019e-3,
            -4.5402305912224966e-2,
        ],
    );
}

#[test]
fn should_compute_silu_in_f32_for_half() {
    assert_half_matches(
        activation::silu,
        &[-15.0, -8.0, 8.0],
        &[
            -4.5885334038843705e-6,
            -2.682801043731825e-3,
            7.997317198956269,
        ],
    );
}

#[test]
fn should_compute_softplus_in_f32_for_half() {
    assert_half_matches(
        |x| activation::softplus(x, 1.0),
        &[-10.0, -8.0, -5.0],
        &[
            4.539889921686465e-5,
            3.3540637289576885e-4,
            6.715348489118068e-3,
        ],
    );
}

#[test]
fn should_compute_mish_in_f32_for_half() {
    assert_half_matches(
        activation::mish,
        &[-8.0, -6.0, 5.0],
        &[
            -2.6832508825465375e-3,
            -1.4854080479424991e-2,
            4.999552077529406,
        ],
    );
}

#[test]
fn should_compute_tanh_in_f32_for_half() {
    assert_half_matches(
        activation::tanh,
        &[0.0009765625, 0.125, 30.0],
        &[9.765621895592603e-4, 0.12435300177159621, 1.0],
    );
}
