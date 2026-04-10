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

/// Approximation of the error function for f32.
/// Uses the Horner form of the approximation from Abramowitz and Stegun.
pub fn erf_f32(x: f32) -> f32 {
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.3275911_f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximation of the error function for f64.
pub fn erf_f64(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorData, Tolerance};

    fn tensor_from_vec(data: Vec<f32>) -> FlexTensor {
        let shape = burn_std::Shape::from(vec![data.len()]);
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    #[test]
    fn test_exp() {
        let tensor = tensor_from_vec(vec![0.0, 1.0, 2.0]);
        let result = exp(tensor);
        let e = std::f32::consts::E;
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0, e, e.powi(2)]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_log() {
        let tensor = tensor_from_vec(vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = log(tensor);
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_sqrt() {
        let tensor = tensor_from_vec(vec![0.0, 1.0, 4.0, 9.0]);
        let result = sqrt(tensor);
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 1.0, 2.0, 3.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_abs() {
        let tensor = tensor_from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
        let result = abs(tensor);
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([3.0, 1.0, 0.0, 1.0, 3.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_sin_cos() {
        let tensor = tensor_from_vec(vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]);

        sin(tensor.clone()).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 1.0, 0.0]),
            Tolerance::absolute(1e-5),
        );

        cos(tensor).into_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0, 0.0, -1.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_tanh() {
        let tensor = tensor_from_vec(vec![-2.0, 0.0, 2.0]);
        let result = tanh(tensor);
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([(-2.0f32).tanh(), 0.0, 2.0f32.tanh()]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_round_floor_ceil() {
        let tensor = tensor_from_vec(vec![-1.5, -0.5, 0.5, 1.5]);

        // Banker's rounding (ties to even): -0.5 -> 0, 0.5 -> 0, -1.5 -> -2, 1.5 -> 2
        round(tensor.clone()).into_data().assert_approx_eq::<f32>(
            &TensorData::from([-2.0, 0.0, 0.0, 2.0]),
            Tolerance::absolute(1e-5),
        );

        floor(tensor.clone()).into_data().assert_approx_eq::<f32>(
            &TensorData::from([-2.0, -1.0, 0.0, 1.0]),
            Tolerance::absolute(1e-5),
        );

        ceil(tensor).into_data().assert_approx_eq::<f32>(
            &TensorData::from([-1.0, 0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_erf() {
        let tensor = tensor_from_vec(vec![0.0, 0.5, 1.0, 2.0]);
        let result = erf(tensor);
        // Expected values from standard erf tables
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.5205, 0.8427, 0.9953]),
            Tolerance::absolute(1e-3),
        );
    }

    // === Non-contiguous tensor tests ===

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, vec![rows, cols]))
    }

    #[test]
    fn test_exp_transposed() {
        // [[0, 1], [2, 3]] transposed -> [[0, 2], [1, 3]]
        // Storage order: [0, 1, 2, 3], but logical order after transpose: [0, 2, 1, 3]
        let tensor = tensor_2d(vec![0.0, 1.0, 2.0, 3.0], 2, 2);
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());

        let e = std::f32::consts::E;
        // exp([0, 2, 1, 3]) = [1.0, e^2, e, e^3]
        exp(transposed).into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![1.0, e * e, e, e * e * e], vec![2, 2]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_sqrt_narrowed() {
        // Original: [1, 4, 9, 16, 25, 36] shape [6]
        // Narrow to middle 4 elements: [4, 9, 16, 25]
        let tensor = tensor_from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let narrowed = tensor.narrow(0, 1, 4);
        assert!(!narrowed.is_contiguous() || narrowed.layout().start_offset() != 0);

        sqrt(narrowed).into_data().assert_approx_eq::<f32>(
            &TensorData::from([2.0, 3.0, 4.0, 5.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_abs_flipped() {
        // Test with negative strides from flip
        // [1, -2, 3, -4] flipped -> [-4, 3, -2, 1]
        let tensor = tensor_from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let flipped = crate::ops::flip::flip(tensor, &[0]);

        // Verify it's using negative strides (zero-copy)
        assert!(flipped.layout().strides()[0] < 0);

        // abs([-4, 3, -2, 1]) = [4, 3, 2, 1]
        abs(flipped).into_data().assert_approx_eq::<f32>(
            &TensorData::from([4.0, 3.0, 2.0, 1.0]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_sqrt_flipped_2d() {
        // [[1, 4], [9, 16]] with axis 0 flipped -> [[9, 16], [1, 4]]
        // sqrt of that -> [[3, 4], [1, 2]]
        let tensor = tensor_2d(vec![1.0, 4.0, 9.0, 16.0], 2, 2);
        let flipped = crate::ops::flip::flip(tensor, &[0]);

        // Axis 0 stride should be negative
        assert!(flipped.layout().strides()[0] < 0);

        // sqrt([[9, 16], [1, 4]]) = [[3, 4], [1, 2]]
        sqrt(flipped).into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_cos_flipped_axis1() {
        // [[0, pi], [pi/2, 3pi/2]] with axis 1 flipped -> [[pi, 0], [3pi/2, pi/2]]
        // cos of that -> [[-1, 1], [0, 0]]
        use std::f32::consts::{FRAC_PI_2, PI};
        let tensor = tensor_2d(vec![0.0, PI, FRAC_PI_2, 3.0 * FRAC_PI_2], 2, 2);
        let flipped = crate::ops::flip::flip(tensor, &[1]);

        // Axis 1 stride should be negative
        assert!(flipped.layout().strides()[1] < 0);

        // cos([[pi, 0], [3pi/2, pi/2]]) = [[-1, 1], [0, 0]]
        cos(flipped).into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![-1.0, 1.0, 0.0, 0.0], vec![2, 2]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_sin_step_sliced() {
        // Step-2 slice creates stride=2 on last dim, which previously broke
        // block_starts() iteration in the unary op path.
        //
        // [0, 1, 2, 3, 4, 5, 6, 7] shape [1, 4]
        // slice(s![.., 0..;2]) -> [0, 2, 4, 6] shape [1, 2] with strides [4, 2]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![1, 8],
        ));

        // Step-2 slice: take even-indexed elements
        let sliced = crate::ops::slice::slice(
            tensor,
            &[
                burn_backend::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                },
                burn_backend::Slice {
                    start: 0,
                    end: None,
                    step: 2,
                },
            ],
        );
        assert_eq!(sliced.layout().shape().to_vec(), vec![1, 4]);
        assert_eq!(sliced.layout().strides()[1], 2); // stride=2 on last dim

        // Verify the sliced data is correct before applying sin
        sliced.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![0.0, 2.0, 4.0, 6.0], vec![1, 4]),
            Tolerance::absolute(1e-6),
        );

        // Apply sin to the step-sliced tensor
        let expected: Vec<f32> = [0.0f32, 2.0, 4.0, 6.0].iter().map(|x| x.sin()).collect();
        sin(sliced).into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![1, 4]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_cos_step_sliced_3d() {
        // 3D tensor with step-2 slice on last dim (the RF-DETR pattern)
        // shape [1, 2, 6] -> slice(s![.., .., 0..;2]) -> shape [1, 2, 3] with stride[2]=2
        let vals: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let tensor = FlexTensor::from_data(TensorData::new(vals, vec![1, 2, 6]));

        let sliced = crate::ops::slice::slice(
            tensor,
            &[
                burn_backend::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                },
                burn_backend::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                },
                burn_backend::Slice {
                    start: 0,
                    end: None,
                    step: 2,
                },
            ],
        );
        assert_eq!(sliced.layout().shape().to_vec(), vec![1, 2, 3]);

        // Even indices from each row: row0 = [0, 1.0, 2.0], row1 = [3.0, 4.0, 5.0]
        sliced.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 2, 3]),
            Tolerance::absolute(1e-6),
        );

        let expected: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|x| x.cos())
            .collect();
        cos(sliced).into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![1, 2, 3]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_log_3d_transposed() {
        // 3D tensor with permuted dimensions
        // Shape [2, 2, 2] -> permute to [2, 2, 2] with different strides
        let e = std::f32::consts::E;
        let data = vec![1.0, e, e * e, e * e * e, 1.0, e, e * e, e * e * e];
        let tensor = FlexTensor::from_data(TensorData::new(data, vec![2, 2, 2]));
        let permuted = tensor.permute(&[2, 0, 1]); // Swap dimensions around
        assert!(!permuted.is_contiguous());

        let result = log(permuted);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // All values should be 0, 1, 2, or 3 depending on permutation
        for &v in &out {
            assert!(v >= -0.01 && v <= 3.01, "unexpected log value: {}", v);
        }
    }

    #[test]
    fn test_round_ties_even_large_float() {
        // Values outside i32 range used to break the manual round_ties_even impl
        let data = vec![2e18_f32, -2e18_f32, f32::MAX, f32::MIN];
        let tensor = tensor_from_vec(data.clone());
        let result = round(tensor);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        for (a, b) in out.iter().zip(data.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
