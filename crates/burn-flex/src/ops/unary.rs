//! Unary tensor operations (exp, log, sqrt, sin, cos, etc.).

use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{BoolDType, Bytes, bf16, f16};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::layout::StridedBlocks;
use crate::{FlexTensor, Layout};

/// Apply a float predicate element-wise, producing a boolean tensor.
///
/// Delegates to [`crate::ops::comparison::make_bool_tensor`] for output
/// construction, so it shares the same `BoolDType` support (Native/U8 only,
/// panics on U32).
pub fn float_predicate<F32P, F64P>(
    tensor: FlexTensor,
    out_dtype: BoolDType,
    f32_pred: F32P,
    f64_pred: F64P,
) -> FlexTensor
where
    F32P: Fn(f32) -> bool + Copy,
    F64P: Fn(f64) -> bool + Copy,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let n = shape.num_elements();

    let result: Vec<u8> = match tensor.dtype() {
        DType::F32 => {
            let s: &[f32] = tensor.storage();
            s[..n].iter().map(|&x| f32_pred(x) as u8).collect()
        }
        DType::F64 => {
            let s: &[f64] = tensor.storage();
            s[..n].iter().map(|&x| f64_pred(x) as u8).collect()
        }
        DType::F16 => {
            let s: &[f16] = tensor.storage();
            s[..n].iter().map(|&x| f32_pred(x.to_f32()) as u8).collect()
        }
        DType::BF16 => {
            let s: &[bf16] = tensor.storage();
            s[..n].iter().map(|&x| f32_pred(x.to_f32()) as u8).collect()
        }
        dt => panic!("float_predicate: expected float dtype, got {:?}", dt),
    };

    crate::ops::comparison::make_bool_tensor(result, shape, out_dtype)
}

/// Apply a unary operation element-wise to a tensor.
pub fn unary_op<F32Op, F64Op>(tensor: FlexTensor, f32_op: F32Op, f64_op: F64Op) -> FlexTensor
where
    F32Op: Fn(f32) -> f32 + Copy,
    F64Op: Fn(f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => unary_op_typed(tensor, f32_op),
        DType::F64 => unary_op_typed(tensor, f64_op),
        DType::F16 => unary_op_typed(tensor, |x: f16| f16::from_f32(f32_op(x.to_f32()))),
        DType::BF16 => unary_op_typed(tensor, |x: bf16| bf16::from_f32(f32_op(x.to_f32()))),
        _ => panic!("unary_op: unsupported dtype {:?}", dtype),
    }
}

/// Generic unary operation for any element type.
fn unary_op_typed<E, Op>(mut tensor: FlexTensor, op: Op) -> FlexTensor
where
    E: burn_backend::Element + bytemuck::Pod,
    Op: Fn(E) -> E,
{
    let n = tensor.layout().num_elements();

    // In-place fast path: unique, contiguous tensor at offset 0
    if tensor.is_unique() && tensor.layout().is_contiguous() && tensor.layout().start_offset() == 0
    {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..n].iter_mut() {
            *x = op(*x);
        }
        return tensor;
    }

    // Allocating path for non-contiguous or offset tensors
    let layout = tensor.layout().clone();
    let src: &[E] = tensor.storage();

    // Check for negative strides (from flip operations)
    let has_negative_strides = layout.strides().iter().any(|&s| s < 0);

    // Fast path: storage exactly matches tensor view (covers transposed tensors)
    // Iterate in storage order (contiguous) and preserve original layout.
    // Only valid when all strides are positive.
    if !has_negative_strides && layout.start_offset() == 0 && src.len() == n {
        let result: Vec<E> = src.iter().map(|&x| op(x)).collect();
        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(bytes, layout, E::dtype());
    }

    // Fallback for negative strides: use StridedIter for correct element order
    if has_negative_strides {
        let result: Vec<E> = crate::strided_index::StridedIter::new(&layout)
            .map(|idx| op(src[idx]))
            .collect();
        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(
            bytes,
            Layout::contiguous(layout.shape().clone()),
            E::dtype(),
        );
    }

    // General path for views/slices with offset or extra storage
    let blocks = layout.strided_blocks();
    let result = match &blocks {
        // Single contiguous block (with offset)
        StridedBlocks::Single { start, len } => {
            src[*start..*start + *len].iter().map(|&x| op(x)).collect()
        }
        // Strided: iterate over contiguous blocks
        StridedBlocks::Multiple {
            block_len,
            num_blocks,
            ..
        } => {
            let block_len = *block_len;
            let num_blocks = *num_blocks;
            let mut result = Vec::with_capacity(n);

            if block_len == 1 {
                for block_start in blocks.block_starts() {
                    result.push(op(src[block_start]));
                }
            } else {
                for block_start in blocks.block_starts() {
                    for i in 0..block_len {
                        result.push(op(src[block_start + i]));
                    }
                }
            }
            debug_assert_eq!(result.len(), num_blocks * block_len);
            result
        }
    };

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(layout.shape().clone()),
        E::dtype(),
    )
}

// ============================================================================
// Specific unary operations
// ============================================================================

/// Exponential: e^x
pub fn exp(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.exp(), |x| x.exp())
}

/// Natural logarithm: ln(x)
pub fn log(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.ln(), |x| x.ln())
}

/// Natural logarithm of (1 + x): ln(1 + x)
pub fn log1p(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.ln_1p(), |x| x.ln_1p())
}

/// Square root
pub fn sqrt(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.sqrt(), |x| x.sqrt())
}

/// Absolute value (float)
pub fn abs(tensor: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    if tensor.dtype() == DType::F32
        && tensor.is_unique()
        && tensor.layout().is_contiguous()
        && tensor.layout().start_offset() == 0
    {
        let n = tensor.layout().num_elements();
        let mut tensor = tensor;
        let storage: &mut [f32] = tensor.storage_mut();
        crate::simd::abs_inplace_f32(&mut storage[..n]);
        return tensor;
    }
    unary_op(tensor, |x| x.abs(), |x| x.abs())
}

/// Absolute value (integer)
pub fn int_abs(tensor: FlexTensor) -> FlexTensor {
    let dtype = tensor.dtype();
    match dtype {
        DType::I64 => unary_op_typed::<i64, _>(tensor, |x| x.wrapping_abs()),
        DType::I32 => unary_op_typed::<i32, _>(tensor, |x| x.wrapping_abs()),
        DType::I16 => unary_op_typed::<i16, _>(tensor, |x| x.wrapping_abs()),
        DType::I8 => unary_op_typed::<i8, _>(tensor, |x| x.wrapping_abs()),
        // Unsigned integers: abs is identity
        DType::U64 | DType::U32 | DType::U16 | DType::U8 => tensor,
        _ => panic!("int_abs: unsupported dtype {:?}", dtype),
    }
}

/// Reciprocal: 1/x
pub fn recip(tensor: FlexTensor) -> FlexTensor {
    #[cfg(feature = "simd")]
    if tensor.dtype() == DType::F32
        && tensor.is_unique()
        && tensor.layout().is_contiguous()
        && tensor.layout().start_offset() == 0
    {
        let n = tensor.layout().num_elements();
        let mut tensor = tensor;
        let storage: &mut [f32] = tensor.storage_mut();
        crate::simd::recip_inplace_f32(&mut storage[..n]);
        return tensor;
    }
    unary_op(tensor, |x| 1.0 / x, |x| 1.0 / x)
}

/// Cosine
pub fn cos(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.cos(), |x| x.cos())
}

/// Sine
pub fn sin(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.sin(), |x| x.sin())
}

/// Tangent
pub fn tan(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.tan(), |x| x.tan())
}

/// Hyperbolic cosine
pub fn cosh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.cosh(), |x| x.cosh())
}

/// Hyperbolic sine
pub fn sinh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.sinh(), |x| x.sinh())
}

/// Hyperbolic tangent
pub fn tanh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.tanh(), |x| x.tanh())
}

/// Inverse cosine (arccos)
pub fn acos(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.acos(), |x| x.acos())
}

/// Inverse hyperbolic cosine
pub fn acosh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.acosh(), |x| x.acosh())
}

/// Inverse sine (arcsin)
pub fn asin(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.asin(), |x| x.asin())
}

/// Inverse hyperbolic sine
pub fn asinh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.asinh(), |x| x.asinh())
}

/// Inverse tangent (arctan)
pub fn atan(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.atan(), |x| x.atan())
}

/// Inverse hyperbolic tangent
pub fn atanh(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.atanh(), |x| x.atanh())
}

/// Round to nearest integer (ties to even / banker's rounding)
pub fn round(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, round_ties_even_f32, round_ties_even_f64)
}

fn round_ties_even_f32(x: f32) -> f32 {
    round_ties_even(x)
}

fn round_ties_even_f64(x: f64) -> f64 {
    round_ties_even(x)
}

/// Round to nearest integer, ties to even (banker's rounding).
///
/// `num_traits::Float::round` rounds ties away from zero. This corrects
/// the halfway case to round to the nearest even integer instead.
///
/// Safety of the `to_i64` path: for f32, values with magnitude >= 2^23
/// have no fractional bits, so `(x - r).abs()` is always 0.0 (never 0.5).
/// For f64, the threshold is 2^52. The halfway check therefore only
/// triggers for values well within i64 range.
fn round_ties_even<F: num_traits::Float + num_traits::ToPrimitive>(x: F) -> F {
    let r = x.round();
    if (x - r).abs() == F::from(0.5).unwrap() {
        // Ties: round to even by checking if r is odd
        match r.to_i64() {
            Some(ri) if ri % 2 != 0 => r - x.signum(),
            _ => r,
        }
    } else {
        r
    }
}

/// Floor (round down)
pub fn floor(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.floor(), |x| x.floor())
}

/// Ceiling (round up)
pub fn ceil(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.ceil(), |x| x.ceil())
}

/// Truncate (round towards zero)
pub fn trunc(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, |x| x.trunc(), |x| x.trunc())
}

/// Error function
pub fn erf(tensor: FlexTensor) -> FlexTensor {
    unary_op(tensor, erf_f32, erf_f64)
}

// ============================================================================
// Error function implementation
// ============================================================================
//
// Delegates to libm for full f32 / f64 precision. The previous Abramowitz
// and Stegun 7.1.26 formula was capped at ~1.5e-7 max absolute error,
// adequate for f32 but well short of f64's ~2.2e-16 precision. libm's erf
// is derived from fdlibm and lands within a few ulps across the full
// range in both precisions.

/// Error function for f32.
pub fn erf_f32(x: f32) -> f32 {
    libm::erff(x)
}

/// Error function for f64.
pub fn erf_f64(x: f64) -> f64 {
    libm::erf(x)
}

// Tests kept here probe flex-internal unary helpers that don't flow through
// the generic FloatTensorOps API. Specifically, the manual round_ties_even
// implementation has a subtle to_i64 conversion path that must not trip on
// values outside i64 range. Plain elementwise unary smokes (exp, log, sqrt,
// abs, sin/cos, tanh, round/floor/ceil, erf) and their stride-through-op
// variants (transposed/flipped/narrowed/sliced) have been migrated to
// burn-backend-tests so they run against every backend. When adding new
// tests, keep them here only if they probe flex-internal helpers
// (erf_f32/f64, round_ties_even); otherwise add them to
// crates/burn-backend-tests/tests/tensor/float/ops/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::TensorData;

    use crate::FlexTensor;

    #[test]
    fn test_round_ties_even_large_float() {
        // Values outside i32 range used to break the manual round_ties_even impl.
        let data = vec![2e18_f32, -2e18_f32, f32::MAX, f32::MIN];
        let tensor = FlexTensor::from_data(TensorData::new(data.clone(), [data.len()]));
        let result = super::round(tensor);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // The previous A&S 7.1.26 erf implementation had ~1.5e-7 max absolute
    // error regardless of arithmetic precision. That was acceptable for
    // f32 but orders of magnitude worse than f64's ~2.2e-16 precision.
    // These tests guard near-ulp accuracy across the domain, including
    // libm's piecewise-rational switch points near |x|=0.84375 and the
    // saturation region around |x|>=6 where erf(x) rounds to ±1.
    // Reference values from Wolfram Alpha / DLMF.
    #[test]
    fn test_erf_f64_is_full_precision() {
        // Reference values computed with mpmath at 25-digit precision.
        let cases = [
            (0.0f64, 0.0f64),
            (1e-10, 1.128379167095512574e-10), // small-x linear regime
            (0.5, 0.5204998778130465377),
            (0.84, 0.7651427114549945347), // just below libm's 0.84375 boundary
            (0.85, 0.7706680576083525324), // just above
            (1.0, 0.8427007929497148693),
            (1.5, 0.9661051464753107271),
            (2.0, 0.9953222650189527342),
            (3.0, 0.9999779095030014145),
            (6.0, 0.9999999999999999785), // near saturation
            (30.0, 1.0),                  // fully saturated
            (-0.5, -0.5204998778130465377),
            (-1.0, -0.8427007929497148693),
            (-3.0, -0.9999779095030014145),
        ];
        for (x, expected) in cases {
            let got = super::erf_f64(x);
            let err = (got - expected).abs();
            assert!(
                err < 1e-14,
                "erf_f64({}) = {} expected {} (err {:e})",
                x,
                got,
                expected,
                err
            );
        }
    }

    // f32 erf must stay accurate to within f32 precision (~6e-8). Covers
    // the same piecewise boundaries as the f64 test.
    #[test]
    fn test_erf_f32_full_f32_precision() {
        let cases = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (0.84, 0.765_142_7),
            (0.85, 0.770_668_06),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_24),
            (6.0, 1.0),
            (-0.5, -0.520_499_9),
            (-1.0, -0.842_700_8),
        ];
        for (x, expected) in cases {
            let got = super::erf_f32(x);
            assert!(
                (got - expected).abs() < 1e-6,
                "erf_f32({}) = {} expected {}",
                x,
                got,
                expected
            );
        }
    }
}
