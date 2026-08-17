//! Quantization accuracy measurement.
//!
//! Reports dequantization error against bits per element, so a scheme can be judged on the
//! accuracy it buys for the memory it costs rather than on either number alone.
//!
//! Run the full sweep with:
//! `cargo test -p burn-backend-tests --features ndarray --test tensor report_quantization_accuracy -- --ignored --nocapture`

use super::*;
use burn_tensor::{
    Shape, TensorData,
    quantization::{
        QuantScheme, QuantValue, ScaleDtype, global_scale_size, params_shape, scale_size,
    },
};

/// Total storage cost of a quantized tensor, per element of the original tensor.
///
/// This is the axis that makes schemes comparable: a smaller block buys accuracy but costs more
/// scales, so only error-at-equal-bits distinguishes a real improvement from a trade.
fn bits_per_element(scheme: &QuantScheme, shape: &Shape) -> f64 {
    let numel = shape.num_elements();
    let num_scales = params_shape(shape, scheme).num_elements();
    scheme.size_bits_value() as f64
        + ((scale_size(scheme.scale_dtype()) * num_scales + global_scale_size(scheme)) * 8) as f64
            / numel as f64
}

/// Deterministic standard-normal samples, so a reported number is reproducible across runs and
/// backends. Box-Muller over a small LCG; quality beyond "plausibly weight-shaped" is not needed.
fn normal_samples(n: usize, std: f32) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next_uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 24 bits, mapped to (0, 1) so the log below never sees zero.
        let bits = (state >> 40) as u32;
        (bits as f32 + 0.5) / (1u32 << 24) as f32
    };

    (0..n)
        .map(|_| {
            let (u1, u2) = (next_uniform(), next_uniform());
            let r = (-2.0 * u1.ln()).sqrt();
            r * (core::f32::consts::TAU * u2).cos() * std
        })
        .collect()
}

/// Relative Frobenius error `||W - Ŵ|| / ||W||`, which is scale free and so comparable across
/// weight magnitudes.
fn relative_error(original: &[f32], dequantized: &[f32]) -> f32 {
    let (mut err_sq, mut ref_sq) = (0.0f64, 0.0f64);
    for (w, w_hat) in original.iter().zip(dequantized) {
        let d = (w - w_hat) as f64;
        err_sq += d * d;
        ref_sq += (*w as f64) * (*w as f64);
    }
    (err_sq.sqrt() / ref_sq.sqrt()) as f32
}

/// Read a tensor back as `f32`, whatever float type the backend is built with.
fn to_f32(tensor: TestTensor<2>) -> Vec<f32> {
    tensor.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

/// Quantize with `scheme`, dequantize, and report the relative error.
///
/// The reference is the tensor as the backend actually holds it, not the `f32` input. Under a
/// narrower float element type the input is rounded on the way in, and charging that to
/// quantization would inflate every measurement by a constant that has nothing to do with the
/// scheme.
fn quantization_error(values: &[f32], shape: [usize; 2], scheme: &QuantScheme) -> f32 {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values.to_vec(), shape), &device);

    let reference = to_f32(tensor.clone());
    let dequantized = to_f32(tensor.quantize_dynamic(scheme).dequantize());

    relative_error(&reference, &dequantized)
}

/// The scale levels a case measures, with the block extent it quantizes on.
#[derive(Clone, Copy, PartialEq)]
enum Levels {
    Tensor,
    Block(u8),
    BlockUnderTensor(u8),
}

fn scheme_for(value: QuantValue, levels: Levels, dtype: ScaleDtype) -> QuantScheme {
    let device = burn_tensor::Device::default();
    let scheme = device.settings().quantization.scheme.with_value(value);

    match levels {
        Levels::Tensor => scheme.per_tensor(dtype),
        Levels::Block(block) => scheme.per_block([block], dtype),
        Levels::BlockUnderTensor(block) => {
            scheme.per_block([block], dtype).per_tensor(ScaleDtype::F32)
        }
    }
}

const SHAPE: [usize; 2] = [64, 64];

/// A scale that underflows the dtype's representable range is the failure the per-tensor scale of
/// a two-level scheme exists to prevent, so it has to be visible here before that scheme is built.
///
/// `Q8S` divides the block maximum by 127, so for weights of a realistic magnitude the block scale
/// lands in e4m3's subnormals or below, where almost no precision is left.
#[test]
fn narrow_scale_dtype_degrades_without_normalization() {
    let values = normal_samples(SHAPE[0] * SHAPE[1], 0.02);
    let levels = Levels::Block(32);

    let exact = quantization_error(
        &values,
        SHAPE,
        &scheme_for(QuantValue::Q8S, levels, ScaleDtype::F32),
    );
    let narrow = quantization_error(
        &values,
        SHAPE,
        &scheme_for(QuantValue::Q8S, levels, ScaleDtype::UE4M3),
    );

    assert!(
        narrow > exact * 2.0,
        "8-bit block scales should degrade badly without a normalizing per-tensor scale, \
         got f32={exact:.4} ue4m3={narrow:.4}"
    );
}

/// At a magnitude where no scale underflows, the two 16-bit dtypes cost nothing measurable while
/// the 8-bit one already costs something. Pinning both directions keeps the harness honest: it
/// has to be sensitive enough to see the 8-bit loss, and not so noisy that it invents a 16-bit one.
///
/// The upper bound on the 8-bit cost is what keeps `scale_to_dtype` rounding up. Rounding a scale
/// to nearest instead lets a block's largest value clip, which measured several times worse.
///
/// Note this is not monotone in scale width. Rounding a scale can land favourably on a given
/// sample, so f16 sitting a hair below f32 is expected and is not a signal.
#[test]
fn error_responds_to_scale_dtype() {
    let values = normal_samples(SHAPE[0] * SHAPE[1], 1.0);
    let levels = Levels::Block(32);

    let error_for =
        |dtype| quantization_error(&values, SHAPE, &scheme_for(QuantValue::Q8S, levels, dtype));
    let (f32_err, f16_err, ue4m3_err) = (
        error_for(ScaleDtype::F32),
        error_for(ScaleDtype::F16),
        error_for(ScaleDtype::UE4M3),
    );

    // Generous on purpose. The measured gap is near zero on the backends checked so far, but this
    // runs on every backend and float element type, and the point is only to separate
    // "indistinguishable" from the 8-bit case below.
    assert!(
        (f16_err - f32_err).abs() / f32_err < 0.20,
        "a 16-bit scale should be indistinguishable from f32, got f32={f32_err} f16={f16_err}"
    );
    assert!(
        ue4m3_err > f32_err,
        "an 8-bit scale should cost some accuracy, got f32={f32_err} ue4m3={ue4m3_err}"
    );
    assert!(
        ue4m3_err < f32_err * 1.3,
        "an 8-bit scale should cost only the coarser step, not a clipped block maximum, \
         got f32={f32_err} ue4m3={ue4m3_err}"
    );
}

#[test]
#[ignore = "reports measurements rather than asserting; run with --ignored --nocapture"]
fn report_quantization_accuracy() {
    let cases = [
        ("tensor", Levels::Tensor),
        ("block16", Levels::Block(16)),
        ("block32", Levels::Block(32)),
        ("block16+f32", Levels::BlockUnderTensor(16)),
        ("block32+f32", Levels::BlockUnderTensor(32)),
    ];
    let dtypes = [
        ("f32", ScaleDtype::F32),
        ("f16", ScaleDtype::F16),
        ("ue4m3", ScaleDtype::UE4M3),
    ];
    let shape = Shape::from(SHAPE);

    for std in [1.0f32, 0.1, 0.02] {
        let values = normal_samples(SHAPE[0] * SHAPE[1], std);

        println!("\nweight std = {std}  (shape {SHAPE:?})");
        println!(
            "{:<13} {:<7} {:>8} {:>12}",
            "levels", "scale", "bits/el", "rel. error"
        );

        for (case_name, levels) in cases {
            for (dtype_name, dtype) in dtypes {
                // A per-tensor scale has nothing to absorb at f32 block scales.
                if matches!(levels, Levels::BlockUnderTensor(_)) && dtype == ScaleDtype::F32 {
                    continue;
                }
                let scheme = scheme_for(QuantValue::Q8S, levels, dtype);
                let error = quantization_error(&values, SHAPE, &scheme);

                println!(
                    "{case_name:<13} {dtype_name:<7} {:>8.3} {error:>12.5}",
                    bits_per_element(&scheme, &shape)
                );
            }
        }
    }
}
