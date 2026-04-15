//! Activation function operations for the Flex backend.
//!
//! Each activation is implemented as a single-pass unary operation,
//! replacing the default multi-op compositions from Burn's trait defaults.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::Scalar;
use burn_backend::ops::{ActivationOps, FloatTensorOps};
use burn_backend::tensor::FloatTensor;
use burn_backend::{DType, TensorMetadata};
use burn_std::{Bytes, bf16, f16};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::ToPrimitive;

use crate::ops::binary::binary_op;
use crate::ops::unary::unary_op;
use crate::{Flex, FlexTensor, Layout};

impl ActivationOps<Flex> for Flex {
    fn relu(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary_op(tensor, |x: f32| x.max(0.0), |x: f64| x.max(0.0))
    }

    fn relu_backward(output: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // grad * (output > 0): zero the gradient where output was zero
        binary_op(
            output,
            grad,
            |out: f32, g| if out > 0.0 { g } else { 0.0 },
            |out: f64, g| if out > 0.0 { g } else { 0.0 },
            None,
        )
    }

    fn leaky_relu(tensor: FloatTensor<Flex>, negative_slope: Scalar) -> FloatTensor<Flex> {
        let ns32 = negative_slope.to_f32().unwrap();
        let ns64 = negative_slope.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| if x >= 0.0 { x } else { ns32 * x },
            move |x: f64| if x >= 0.0 { x } else { ns64 * x },
        )
    }

    fn prelu(tensor: FloatTensor<Flex>, alpha: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // x if x >= 0, alpha * x otherwise
        binary_op(
            tensor,
            alpha,
            |x: f32, a| if x >= 0.0 { x } else { a * x },
            |x: f64, a| if x >= 0.0 { x } else { a * x },
            None,
        )
    }

    fn gelu(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        unary_op(
            tensor,
            move |x: f32| 0.5 * x * (1.0 + erf_f32(x / sqrt2_f32)),
            move |x: f64| 0.5 * x * (1.0 + erf_f64(x / sqrt2_f64)),
        )
    }

    fn gelu_backward(x: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // d/dx[gelu(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        let inv_sqrt_2pi_f32: f32 = 1.0 / (2.0 * core::f32::consts::PI).sqrt();
        let inv_sqrt_2pi_f64: f64 = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
        binary_op(
            x,
            grad,
            move |x: f32, g| {
                let cdf = 0.5 * (1.0 + erf_f32(x / sqrt2_f32));
                let pdf = inv_sqrt_2pi_f32 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            move |x: f64, g| {
                let cdf = 0.5 * (1.0 + erf_f64(x / sqrt2_f64));
                let pdf = inv_sqrt_2pi_f64 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            None,
        )
    }

    fn sigmoid(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary_op(tensor, sigmoid_f32, sigmoid_f64)
    }

    fn sigmoid_backward(output: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // grad * output * (1 - output)
        binary_op(
            output,
            grad,
            |s: f32, g| g * s * (1.0 - s),
            |s: f64, g| g * s * (1.0 - s),
            None,
        )
    }

    fn hard_sigmoid(tensor: FloatTensor<Flex>, alpha: Scalar, beta: Scalar) -> FloatTensor<Flex> {
        let alpha32 = alpha.to_f32().unwrap();
        let beta32 = beta.to_f32().unwrap();
        let alpha64 = alpha.to_f64().unwrap();
        let beta64 = beta.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| (alpha32 * x + beta32).clamp(0.0, 1.0),
            move |x: f64| (alpha64 * x + beta64).clamp(0.0, 1.0),
        )
    }

    fn log_sigmoid(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // Numerically stable: -softplus(-x) = -log(1 + exp(-x))
        // For x >= 0: -log(1 + exp(-x))  (standard form, exp(-x) is small)
        // For x < 0: x - log(1 + exp(x))  (avoids exp of large positive)
        unary_op(
            tensor,
            |x: f32| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
            |x: f64| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
        )
    }

    fn log_sigmoid_backward(x: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // d/dx[log_sigmoid(x)] = sigmoid(-x) * (-1) * (-1) = 1 - sigmoid(x) = sigmoid(-x)
        // So: grad * sigmoid(-x)
        binary_op(
            x,
            grad,
            |x: f32, g| g * sigmoid_f32(-x),
            |x: f64, g| g * sigmoid_f64(-x),
            None,
        )
    }

    fn softmax(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        softmax(tensor, dim)
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ============================================================================
// Fused softmax
// ============================================================================
//
// `ActivationOps` does not currently expose a `softmax` hook, so
// `burn_tensor::activation::softmax` falls back to a 5-op decomposition
// (`max_dim`/`sub`/`exp`/`sum_dim`/`div`). This module provides a fused
// alternative users can opt into directly.

/// Fused softmax along `dim`.
///
/// Three-pass row-wise algorithm (max, exp+sum, normalize) keeping each row
/// cache-hot. Rows are processed in parallel via rayon. For axes other than
/// the last, the tensor is permuted to put `dim` last, the fused kernel runs,
/// and the result is permuted back (both permutes are metadata-only; the
/// fused kernel's internal `to_contiguous` materializes the permuted layout
/// once).
///
/// # Panics
///
/// * If `dim` is out of range for `input`.
/// * If `input`'s dtype is not one of `f32`/`f64`/`f16`/`bf16`.
pub fn softmax(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
    let rank = tensor.shape().num_dims();
    assert!(
        dim < rank,
        "softmax dim {} out of range for rank {}",
        dim,
        rank
    );

    if dim != rank - 1 {
        let swapped = Flex::float_swap_dims(tensor, dim, rank - 1);
        let normed = softmax_last(swapped);
        return Flex::float_swap_dims(normed, dim, rank - 1);
    }

    softmax_last(tensor)
}

fn softmax_last(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
    let tensor = tensor.to_contiguous();
    match tensor.dtype() {
        DType::F32 => softmax_last_f32(tensor),
        DType::F64 => softmax_last_f64(tensor),
        DType::F16 => softmax_last_f16(tensor),
        DType::BF16 => softmax_last_bf16(tensor),
        dtype => panic!("softmax: unsupported dtype {:?}", dtype),
    }
}

fn softmax_last_f32(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape().clone();
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[f32] = tensor.storage();
    let n = input.len();

    // Zero-initialize the output. The previous implementation used
    // `Vec::with_capacity` + `spare_capacity_mut` + a raw-pointer cast to
    // `&mut [f32]` to skip the memset, but forming a `&mut [f32]` over
    // uninitialized memory violates Rust's validity invariant (references
    // must point to initialized values of the correct type) even if every
    // element is written before it is read. The sound zero-memset
    // alternative would require threading `&mut [MaybeUninit<f32>]` through
    // the row kernel, which does not compose with macerator's `#[with_simd]`
    // signature. The memset is a streaming write on a bandwidth-bound
    // kernel, so the overhead is small (~10% at the largest bench shape)
    // and the fused path remains well ahead of decomposed and candle.
    let mut output: Vec<f32> = vec![0.0; n];
    let out_slice = output.as_mut_slice();

    // Row-parallel via rayon: one macerator dispatch per chunk of rows,
    // amortized over all rows in the chunk.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * last;
        out_slice
            .par_chunks_mut(chunk_elems)
            .zip(input.par_chunks(chunk_elems))
            .for_each(|(o, i)| softmax_rows_f32(i, o, last));
    }
    #[cfg(not(feature = "rayon"))]
    {
        softmax_rows_f32(input, out_slice, last);
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F32,
    )
}

/// Row sweep for f32 softmax. With the `simd` feature, delegates to the
/// `#[macerator::with_simd]` SIMD kernel (one dispatch per chunk of rows,
/// amortized over all rows in the chunk). Without `simd`, uses a scalar
/// row kernel.
#[inline]
fn softmax_rows_f32(input: &[f32], output: &mut [f32], row_len: usize) {
    // Release-mode invariant checks. These run once per chunk of rows
    // (dozens of times per call, not per-element), so the overhead is
    // unmeasurable against the kernel work. A debug-only check would
    // silently pass a short final chunk to the row kernel on release
    // builds if a future refactor broke the row alignment at the call
    // site, yielding wrong softmax output with no panic.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % row_len, 0);
    #[cfg(feature = "simd")]
    softmax_rows_f32_simd(input, output, row_len);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
            softmax_row_f32_scalar(in_row, out_row);
        }
    }
}

#[cfg(feature = "simd")]
#[macerator::with_simd]
fn softmax_rows_f32_simd<S: macerator::Simd>(input: &[f32], output: &mut [f32], row_len: usize) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % row_len, 0);
    for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
        softmax_row_f32_simd::<S>(in_row, out_row);
    }
}

/// Scalar fallback row kernel for f32 softmax when the `simd` feature is
/// disabled. Uses the same 3-pass algorithm as the SIMD path; LLVM
/// autovectorizes the max-reduce and normalize loops on most targets.
#[cfg(not(feature = "simd"))]
#[inline]
fn softmax_row_f32_scalar(input: &[f32], output: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0f32;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max_val).exp();
        output[i] = e;
        sum += e;
    }
    let inv = 1.0f32 / sum;
    for x in output.iter_mut() {
        *x *= inv;
    }
}

/// Inner row kernel for a single softmax row. `#[inline(always)]` so it
/// inlines into `softmax_rows_f32_simd`'s loop body for each monomorphized S,
/// avoiding a per-row call boundary.
#[cfg(feature = "simd")]
#[inline(always)]
fn softmax_row_f32_simd<S: macerator::Simd>(input: &[f32], output: &mut [f32]) {
    use macerator::{Scalar, vload_unaligned, vstore_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let len = input.len();
    let simd_len = len / lanes * lanes;

    // Pass 1: row max for numerical stability.
    // SIMD max-reduction across the row, scalar tail.
    let (mut max_val, tail_start) = if simd_len >= lanes {
        let mut max_vec = unsafe { vload_unaligned::<S, _>(input.as_ptr()) };
        let mut j = lanes;
        while j < simd_len {
            let v = unsafe { vload_unaligned::<S, _>(input.as_ptr().add(j)) };
            max_vec = max_vec.max(v);
            j += lanes;
        }
        (max_vec.reduce_max(), simd_len)
    } else {
        (f32::NEG_INFINITY, 0)
    };
    for &x in &input[tail_start..] {
        if x > max_val {
            max_val = x;
        }
    }

    // Pass 2: compute exp(x - max), store in output, accumulate sum.
    // Scalar exp (no SIMD exp in macerator). This pass is the one that
    // actually does memory reads + writes on the whole row, so scalar
    // here still lands us at memory bandwidth.
    let mut sum = 0.0f32;
    for idx in 0..len {
        let e = (input[idx] - max_val).exp();
        output[idx] = e;
        sum += e;
    }

    // Pass 3: normalize.
    // SIMD splat + multiply, scalar tail.
    let inv = 1.0f32 / sum;
    let inv_vec = inv.splat::<S>();
    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(output.as_ptr().add(i));
            vstore_unaligned::<S, _>(output.as_mut_ptr().add(i), v * inv_vec);
        }
        i += lanes;
    }
    for x in &mut output[i..] {
        *x *= inv;
    }
}

// f64, f16, bf16 softmax share the same row-parallel dispatcher shell and
// differ only in their row kernel (native f64 vs via-f32 for half
// precision). Generated via macros to keep the three variants in lockstep.
// Only f32 has a dedicated SIMD fast path above.

macro_rules! softmax_last_dtype {
    ($fn_name:ident, $T:ty, $zero:expr, $dtype:expr, $row_fn:ident) => {
        fn $fn_name(tensor: FlexTensor) -> FlexTensor {
            let shape = tensor.layout().shape().clone();
            let last = *shape.last().expect("softmax: empty shape");
            if last == 0 {
                return tensor;
            }
            let input: &[$T] = tensor.storage();
            let mut output: Vec<$T> = vec![$zero; input.len()];

            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                output
                    .par_chunks_mut(last)
                    .zip(input.par_chunks(last))
                    .for_each(|(o, i)| $row_fn(i, o));
            }
            #[cfg(not(feature = "rayon"))]
            {
                for (i, o) in input.chunks(last).zip(output.chunks_mut(last)) {
                    $row_fn(i, o);
                }
            }

            FlexTensor::new(Bytes::from_elems(output), Layout::contiguous(shape), $dtype)
        }
    };
}

/// Half-precision softmax row kernel. Accumulates in f32 for numerical
/// stability and converts back to the target type at each write. This
/// double-rounds across passes 2 and 3; acceptable for half precision. An
/// f32 scratch buffer would remove the double rounding at the cost of a
/// per-row allocation.
macro_rules! softmax_row_half {
    ($fn_name:ident, $T:ty) => {
        #[inline]
        fn $fn_name(input: &[$T], output: &mut [$T]) {
            let mut max_val = f32::NEG_INFINITY;
            for &x in input {
                let xf = x.to_f32();
                if xf > max_val {
                    max_val = xf;
                }
            }
            let mut sum = 0.0f32;
            for (i, &x) in input.iter().enumerate() {
                let e = (x.to_f32() - max_val).exp();
                output[i] = <$T>::from_f32(e);
                sum += e;
            }
            let inv = 1.0f32 / sum;
            for x in output.iter_mut() {
                *x = <$T>::from_f32(x.to_f32() * inv);
            }
        }
    };
}

#[inline]
fn softmax_row_f64(input: &[f64], output: &mut [f64]) {
    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0f64;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max_val).exp();
        output[i] = e;
        sum += e;
    }
    let inv = 1.0f64 / sum;
    for x in output.iter_mut() {
        *x *= inv;
    }
}

softmax_row_half!(softmax_row_f16, f16);
softmax_row_half!(softmax_row_bf16, bf16);

softmax_last_dtype!(softmax_last_f64, f64, 0.0f64, DType::F64, softmax_row_f64);
softmax_last_dtype!(
    softmax_last_f16,
    f16,
    f16::from_f32(0.0),
    DType::F16,
    softmax_row_f16
);
softmax_last_dtype!(
    softmax_last_bf16,
    bf16,
    bf16::from_f32(0.0),
    DType::BF16,
    softmax_row_bf16
);

// ============================================================================
// Fused layer_norm
// ============================================================================
//
// `burn::nn::LayerNorm::forward` decomposes into ~6 primitive tensor ops
// with intermediate allocations, and there is no backend trait hook for
// layer_norm. This module provides a fused alternative users can opt into
// directly. Two-pass row kernel (sum+sumsq sweep, then normalize+affine
// sweep), both vectorized via macerator.

/// Fused layer normalization along the last axis.
///
/// Applies `y = ((x - mean) / sqrt(var + eps)) * gamma + beta`, where
/// `mean` and `var` are computed per row along the last axis of `input`.
/// `gamma` and `beta` are 1-D tensors of length `input.shape()[-1]`;
/// `beta` is optional (set to `None` for a bias-free layer norm).
///
/// Two-pass row kernel (mean/variance via a single sum+sum-of-squares
/// sweep, then one normalize+affine sweep). Both passes are SIMD via
/// macerator; each row stays cache-hot across both passes.
///
/// Supports `f32` (SIMD-vectorized), `f64` (scalar + LLVM autovec), and
/// `f16`/`bf16` (via an f32 cast-fuse-cast shell; the f32 row kernel
/// already accumulates in f32, so this matches the precision a
/// half-precision-native kernel would produce).
///
/// # Panics
///
/// * If `input`'s dtype is not one of `f32`/`f64`/`f16`/`bf16`.
/// * If `input` has rank 0.
/// * If `gamma` (or `beta`, when present) is not a 1-D tensor of length
///   equal to the last dim of `input`.
pub fn layer_norm(
    input: FloatTensor<Flex>,
    gamma: FloatTensor<Flex>,
    beta: Option<FloatTensor<Flex>>,
    epsilon: f64,
) -> FloatTensor<Flex> {
    let rank = input.shape().num_dims();
    assert!(rank >= 1, "layer_norm: input must have at least one dim");
    // Keep gamma/beta dtypes aligned with the input. The half-precision path
    // (see `layer_norm_via_f32`) ultimately accesses storage using the input's
    // element type, and a mismatch would panic there; reject it up front with
    // a clearer layer_norm-specific error message.
    assert_eq!(
        gamma.dtype(),
        input.dtype(),
        "layer_norm: gamma dtype {:?} does not match input dtype {:?}",
        gamma.dtype(),
        input.dtype(),
    );
    if let Some(ref b) = beta {
        assert_eq!(
            b.dtype(),
            input.dtype(),
            "layer_norm: beta dtype {:?} does not match input dtype {:?}",
            b.dtype(),
            input.dtype(),
        );
    }
    let input = input.to_contiguous();
    let gamma = gamma.to_contiguous();
    let beta = beta.map(|b| b.to_contiguous());

    let d_model = *input
        .layout()
        .shape()
        .last()
        .expect("layer_norm: empty shape");
    // Validate rank + length explicitly rather than just last-dim == d_model.
    // A gamma shaped like `[2, d_model]` would pass a last-dim check but
    // has 2*d_model elements, which would index the wrong data in the row
    // kernel (caught by an inner assert, but with a confusing message).
    let gamma_shape = gamma.layout().shape();
    assert!(
        gamma_shape.len() == 1 && gamma_shape[0] == d_model,
        "layer_norm: gamma must be a 1-D tensor of length equal to last dim of input \
         (got shape {:?}, expected [{}])",
        gamma_shape,
        d_model,
    );
    if let Some(ref b) = beta {
        let beta_shape = b.layout().shape();
        assert!(
            beta_shape.len() == 1 && beta_shape[0] == d_model,
            "layer_norm: beta must be a 1-D tensor of length equal to last dim of input \
             (got shape {:?}, expected [{}])",
            beta_shape,
            d_model,
        );
    }

    match input.dtype() {
        DType::F32 => layer_norm_f32(input, gamma, beta, epsilon as f32),
        DType::F64 => layer_norm_f64(input, gamma, beta, epsilon),
        DType::F16 => {
            layer_norm_via_f32::<f16>(input, gamma, beta, epsilon, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => {
            layer_norm_via_f32::<bf16>(input, gamma, beta, epsilon, bf16::to_f32, bf16::from_f32)
        }
        dtype => panic!("burn_flex::layer_norm: unsupported dtype {:?}", dtype),
    }
}

fn layer_norm_via_f32<E: burn_backend::Element + bytemuck::Pod + Copy>(
    input: FlexTensor,
    gamma: FlexTensor,
    beta: Option<FlexTensor>,
    epsilon: f64,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    let input_f32 = crate::ops::module::cast_to_f32::<E>(input, to_f32);
    let gamma_f32 = crate::ops::module::cast_to_f32::<E>(gamma, to_f32);
    let beta_f32 = beta.map(|b| crate::ops::module::cast_to_f32::<E>(b, to_f32));
    let out = layer_norm_f32(input_f32, gamma_f32, beta_f32, epsilon as f32);
    crate::ops::module::cast_from_f32::<E>(out, from_f32)
}

/// Fused f64 layer_norm. The Welford mean/variance pass is serial (the
/// mean update on iteration `k` depends on iteration `k-1`); the
/// normalize+affine pass autovectorizes on targets with f64 SIMD. A
/// macerator f64 path can be added if profiling shows it matters.
fn layer_norm_f64(
    input: FlexTensor,
    gamma: FlexTensor,
    beta: Option<FlexTensor>,
    epsilon: f64,
) -> FlexTensor {
    let shape = input.layout().shape().clone();
    let d_model = *shape.last().expect("layer_norm: empty shape");
    if d_model == 0 {
        return input;
    }
    let input_data: &[f64] = input.storage();
    let gamma_data: &[f64] = gamma.storage();
    let beta_data: Option<&[f64]> = beta.as_ref().map(|b| b.storage());
    let mut output: Vec<f64> = vec![0.0; input_data.len()];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * d_model;
        match beta_data {
            Some(beta_slice) => {
                output
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f64_with_beta(
                            i, o, gamma_data, beta_slice, d_model, epsilon,
                        );
                    });
            }
            None => {
                output
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f64_no_beta(i, o, gamma_data, d_model, epsilon);
                    });
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        match beta_data {
            Some(beta_slice) => layer_norm_rows_f64_with_beta(
                input_data,
                output.as_mut_slice(),
                gamma_data,
                beta_slice,
                d_model,
                epsilon,
            ),
            None => layer_norm_rows_f64_no_beta(
                input_data,
                output.as_mut_slice(),
                gamma_data,
                d_model,
                epsilon,
            ),
        }
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F64,
    )
}

#[inline]
fn layer_norm_rows_f64_with_beta(
    input: &[f64],
    output: &mut [f64],
    gamma: &[f64],
    beta: &[f64],
    d_model: usize,
    epsilon: f64,
) {
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        let (mean, inv_std) = welford_f64(in_row, epsilon);
        for (i, &x) in in_row.iter().enumerate() {
            out_row[i] = (x - mean) * (inv_std * gamma[i]) + beta[i];
        }
    }
}

#[inline]
fn layer_norm_rows_f64_no_beta(
    input: &[f64],
    output: &mut [f64],
    gamma: &[f64],
    d_model: usize,
    epsilon: f64,
) {
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        let (mean, inv_std) = welford_f64(in_row, epsilon);
        for (i, &x) in in_row.iter().enumerate() {
            out_row[i] = (x - mean) * (inv_std * gamma[i]);
        }
    }
}

#[inline]
fn welford_f64(row: &[f64], epsilon: f64) -> (f64, f64) {
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (k, &x) in row.iter().enumerate() {
        let n_k = (k + 1) as f64;
        let delta = x - mean;
        mean += delta / n_k;
        m2 += delta * (x - mean);
    }
    let var = m2 / row.len() as f64;
    (mean, 1.0f64 / (var + epsilon).sqrt())
}

fn layer_norm_f32(
    input: FlexTensor,
    gamma: FlexTensor,
    beta: Option<FlexTensor>,
    epsilon: f32,
) -> FlexTensor {
    let shape = input.layout().shape().clone();
    let d_model = *shape.last().expect("layer_norm: empty shape");
    if d_model == 0 {
        return input;
    }

    let input_data: &[f32] = input.storage();
    let gamma_data: &[f32] = gamma.storage();
    let beta_data: Option<&[f32]> = beta.as_ref().map(|b| b.storage());

    let n = input_data.len();
    // See softmax_last_f32 for the rationale on zero-init instead of
    // `spare_capacity_mut` + `&mut [f32]` cast: the latter creates a
    // reference to uninitialized f32 values, which is UB under Rust's
    // aliasing model even with no intervening read.
    let mut output: Vec<f32> = vec![0.0; n];
    let out_slice = output.as_mut_slice();

    // `#[macerator::with_simd]` can't auto-lifetime through
    // `Option<&[T]>`, so we dispatch two separate monomorphized
    // versions, one with beta and one without. Both call into the
    // same shared row kernel.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        const ROWS_PER_TASK: usize = 64;
        let chunk_elems = ROWS_PER_TASK * d_model;
        match beta_data {
            Some(beta_slice) => {
                out_slice
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f32_with_beta(
                            i, o, gamma_data, beta_slice, d_model, epsilon,
                        );
                    });
            }
            None => {
                out_slice
                    .par_chunks_mut(chunk_elems)
                    .zip(input_data.par_chunks(chunk_elems))
                    .for_each(|(o, i)| {
                        layer_norm_rows_f32_no_beta(i, o, gamma_data, d_model, epsilon);
                    });
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        match beta_data {
            Some(beta_slice) => layer_norm_rows_f32_with_beta(
                input_data, out_slice, gamma_data, beta_slice, d_model, epsilon,
            ),
            None => {
                layer_norm_rows_f32_no_beta(input_data, out_slice, gamma_data, d_model, epsilon)
            }
        }
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::F32,
    )
}

/// Row sweep for f32 layer_norm with bias. Delegates to the SIMD kernel
/// when the `simd` feature is enabled; otherwise uses a scalar row loop.
#[inline]
fn layer_norm_rows_f32_with_beta(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    // Release-mode invariant checks; see softmax_rows_f32 for rationale.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % d_model, 0);
    assert_eq!(gamma.len(), d_model);
    assert_eq!(beta.len(), d_model);
    #[cfg(feature = "simd")]
    layer_norm_rows_f32_with_beta_simd(input, output, gamma, beta, d_model, epsilon);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
            layer_norm_row_f32_scalar(in_row, out_row, gamma, Some(beta), epsilon);
        }
    }
}

/// Row sweep for f32 layer_norm without bias.
#[inline]
fn layer_norm_rows_f32_no_beta(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    // Release-mode invariant checks; see softmax_rows_f32 for rationale.
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % d_model, 0);
    assert_eq!(gamma.len(), d_model);
    #[cfg(feature = "simd")]
    layer_norm_rows_f32_no_beta_simd(input, output, gamma, d_model, epsilon);
    #[cfg(not(feature = "simd"))]
    {
        for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
            layer_norm_row_f32_scalar(in_row, out_row, gamma, None, epsilon);
        }
    }
}

/// Scalar fallback row kernel for layer_norm when the `simd` feature is
/// disabled. Two-pass algorithm matching the SIMD version (sum+sumsq,
/// then normalize+affine).
#[cfg(not(feature = "simd"))]
#[inline]
fn layer_norm_row_f32_scalar(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    epsilon: f32,
) {
    // Welford's online algorithm for mean and variance, rather than the
    // `sumsq / n - mean * mean` identity the SIMD path uses. The identity
    // is vulnerable to catastrophic cancellation when the two terms are
    // close in magnitude (large mean relative to variance). Welford's
    // single-pass formulation avoids that by tracking the running mean
    // and accumulating squared deviations from it. The scalar path is
    // the contract used when `simd` is disabled, so we prefer numerical
    // stability over bit-for-bit match with the SIMD tree reduction.
    let len = input.len();
    let mut mean = 0.0f32;
    let mut m2 = 0.0f32;
    for (k, &x) in input.iter().enumerate() {
        let n_k = (k + 1) as f32;
        let delta = x - mean;
        mean += delta / n_k;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let var = m2 / len as f32;
    let inv_std = 1.0f32 / (var + epsilon).sqrt();
    for (i, &x) in input.iter().enumerate() {
        let scale = inv_std * gamma[i];
        let normed = (x - mean) * scale;
        output[i] = match beta {
            Some(b) => normed + b[i],
            None => normed,
        };
    }
}

/// SIMD-dispatched row sweep for f32 layer_norm with bias (beta). One
/// macerator dispatch per chunk of rows, amortized over the whole chunk.
#[cfg(feature = "simd")]
#[macerator::with_simd]
fn layer_norm_rows_f32_with_beta_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    debug_assert_eq!(beta.len(), d_model);
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        layer_norm_row_f32_simd::<S>(in_row, out_row, gamma, Some(beta), epsilon);
    }
}

/// SIMD-dispatched row sweep for f32 layer_norm without bias.
#[cfg(feature = "simd")]
#[macerator::with_simd]
fn layer_norm_rows_f32_no_beta_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    d_model: usize,
    epsilon: f32,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % d_model, 0);
    debug_assert_eq!(gamma.len(), d_model);
    for (in_row, out_row) in input.chunks(d_model).zip(output.chunks_mut(d_model)) {
        layer_norm_row_f32_simd::<S>(in_row, out_row, gamma, None, epsilon);
    }
}

/// Single-row layer_norm kernel. Two vectorized passes.
#[cfg(feature = "simd")]
#[inline(always)]
fn layer_norm_row_f32_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    epsilon: f32,
) {
    use macerator::{Scalar, vload_unaligned, vstore_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let len = input.len();
    let simd_len = len / lanes * lanes;

    // Pass 1: compute sum and sum-of-squares in one sweep, then derive
    // mean and variance. Two independent SIMD accumulators (sum, sumsq)
    // expose ILP to the two FMA ports.
    let (sum, sumsq) = if simd_len >= lanes {
        let mut acc_sum = 0.0f32.splat::<S>();
        let mut acc_sumsq = 0.0f32.splat::<S>();
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let v = vload_unaligned::<S, _>(input.as_ptr().add(i));
                acc_sum += v;
                // acc_sumsq += v * v; Vector::mul_add(self, a, b) = self*a + b,
                // so v.mul_add(v, acc_sumsq) = v*v + acc_sumsq.
                acc_sumsq = v.mul_add(v, acc_sumsq);
            }
            i += lanes;
        }
        let mut s = acc_sum.reduce_add();
        let mut sq = acc_sumsq.reduce_add();
        for &x in &input[simd_len..] {
            s += x;
            sq += x * x;
        }
        (s, sq)
    } else {
        let mut s = 0.0f32;
        let mut sq = 0.0f32;
        for &x in input {
            s += x;
            sq += x * x;
        }
        (s, sq)
    };

    let n = len as f32;
    let mean = sum / n;
    // Biased variance: E[x^2] - E[x]^2. Matches burn::nn::LayerNorm which
    // uses var_mean_bias (the biased estimator) rather than Bessel's
    // correction.
    let var = (sumsq / n) - mean * mean;
    let inv_std = 1.0f32 / (var + epsilon).sqrt();

    // Pass 2: normalize and affine transform.
    //   out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i]
    // mean_vec and inv_std_vec are hoisted outside the loop (one splat
    // each per row). gamma and beta are read once per element; both
    // fit in L1 and are shared across all rows within a rayon chunk.
    let mean_vec = mean.splat::<S>();
    let inv_std_vec = inv_std.splat::<S>();
    let mut i = 0;
    while i < simd_len {
        unsafe {
            let x = vload_unaligned::<S, _>(input.as_ptr().add(i));
            let g = vload_unaligned::<S, _>(gamma.as_ptr().add(i));
            // scale = inv_std * g
            let scale = inv_std_vec * g;
            // centered = x - mean
            let centered = x - mean_vec;
            // out = centered * scale  (+ beta if present)
            let normed = centered * scale;
            let out = if let Some(b) = beta {
                let b_vec = vload_unaligned::<S, _>(b.as_ptr().add(i));
                normed + b_vec
            } else {
                normed
            };
            vstore_unaligned::<S, _>(output.as_mut_ptr().add(i), out);
        }
        i += lanes;
    }
    // Scalar tail
    while i < len {
        let centered = input[i] - mean;
        let normed = centered * inv_std * gamma[i];
        output[i] = match beta {
            Some(b) => normed + b[i],
            None => normed,
        };
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use burn_backend::Tolerance;
    use burn_tensor::{Tensor, TensorData, activation};

    use crate::Flex;

    #[test]
    fn test_relu() {
        let t: Tensor<Flex, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::relu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_sigmoid() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // sigmoid(-10) ~ 0, sigmoid(0) = 0.5, sigmoid(10) ~ 1
        activation::sigmoid(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.5, 1.0]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_gelu() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-3.0f32, 0.0, 3.0], &Default::default());
        // gelu(0) = 0, gelu(-3) ~ -0.004, gelu(3) ~ 2.996
        activation::gelu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 3.0]),
            Tolerance::absolute(0.01),
        );
    }

    #[test]
    fn test_leaky_relu() {
        let t: Tensor<Flex, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::leaky_relu(t, 0.01)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-0.02, -0.01, 0.0, 1.0, 2.0]),
                Tolerance::absolute(1e-6),
            );
    }

    #[test]
    fn test_softmax_1d() {
        use burn_tensor::TensorPrimitive;
        // softmax([1, 2, 3]) should equal the reference impl
        let t: Tensor<Flex, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());
        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let result = crate::ops::activation::softmax(primitive, 0);
        let result: Tensor<Flex, 1> = Tensor::from_primitive(TensorPrimitive::Float(result));
        // e^1=2.7183, e^2=7.389, e^3=20.0855, sum=30.193
        // normalized: 0.09003, 0.24473, 0.66524
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.09003, 0.24473, 0.66524]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_2d_last_axis() {
        use burn_tensor::TensorPrimitive;
        // Cross-check against burn_tensor::activation::softmax on the same input
        let data = [[-1.0f32, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]];
        let t: Tensor<Flex, 2> = Tensor::from_data(data, &Default::default());
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_layer_norm_2d_with_beta() {
        use burn_tensor::TensorPrimitive;
        // Reference: layer_norm([2, 4]) along last axis.
        // Row 0: [1, 2, 3, 4], mean=2.5, var=1.25, inv_std=1/sqrt(1.25+1e-5)
        // normalized row 0 ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
        // gamma = [1, 1, 1, 1], beta = [0, 0, 0, 0] → same as normalized.
        // Row 1: [5, 6, 7, 8], mean=6.5, var=1.25 → same normalized values.
        let t: Tensor<Flex, 2> = Tensor::from_data(
            [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            &Default::default(),
        );
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data([1.0f32, 1.0, 1.0, 1.0], &Default::default());
        let beta: Tensor<Flex, 1> = Tensor::from_data([0.0f32; 4], &Default::default());

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let out = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), 1e-5);
        let out: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(out));

        let expected = [
            [-1.3416408, -0.4472136, 0.4472136, 1.3416408],
            [-1.3416408, -0.4472136, 0.4472136, 1.3416408],
        ];
        out.into_data()
            .assert_approx_eq::<f32>(&TensorData::from(expected), Tolerance::absolute(1e-4));
    }

    #[test]
    fn test_layer_norm_with_affine() {
        // gamma/beta should scale and shift the normalized output.
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], &Default::default());
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data([2.0f32, 0.5, 1.0, 3.0], &Default::default());
        let beta: Tensor<Flex, 1> =
            Tensor::from_data([1.0f32, -1.0, 0.0, 2.0], &Default::default());

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let out = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), 1e-5);
        let out: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(out));

        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // affine:
        //   [0] = -1.3416 * 2.0 + 1.0 = -1.6833
        //   [1] = -0.4472 * 0.5 - 1.0 = -1.2236
        //   [2] =  0.4472 * 1.0 + 0.0 =  0.4472
        //   [3] =  1.3416 * 3.0 + 2.0 =  6.0249
        out.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[-1.6833, -1.2236, 0.4472, 6.0249]]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_layer_norm_no_beta() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], &Default::default());
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data([1.0f32, 1.0, 1.0, 1.0], &Default::default());

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        // no beta
        let out = crate::ops::activation::layer_norm(t_prim, g_prim, None, 1e-5);
        let out: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(out));

        out.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[-1.3416408, -0.4472136, 0.4472136, 1.3416408]]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_3d_attention_shape() {
        // wav2vec2-like attention scores [heads, seq_q, seq_k], softmax over seq_k.
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 3> = Tensor::from_data(
            [
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            &Default::default(),
        );
        let reference = activation::softmax(t.clone(), 2);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 2);
        let fused: Tensor<Flex, 3> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_softmax_simd_body_row() {
        // Row length of 32 ensures the SIMD body of softmax_row_f32_simd runs
        // on every supported target: NEON (lanes=4), AVX2 (lanes=8), AVX-512
        // (lanes=16), SIMD128 (lanes=4). Earlier tests use rows of length 3-4
        // which leaves the SIMD body at zero iterations on AVX2+.
        use burn_tensor::{Tensor, TensorData, TensorPrimitive};
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let t: Tensor<Flex, 2> =
            Tensor::from_data(TensorData::new(data, [1, 32]), &Default::default());
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_softmax_multi_chunk_rayon() {
        // 100 rows > ROWS_PER_TASK (64), so the rayon par_chunks path
        // produces at least two tasks. Combined with d_model=16 this also
        // exercises the SIMD body with a row length that doesn't divide
        // evenly on AVX-512 (lanes=16 leaves simd_len=16 and a zero tail;
        // NEON lanes=4 leaves simd_len=16 and a zero tail; on AVX2 lanes=8
        // same story).
        use burn_tensor::{Tensor, TensorData, TensorPrimitive};
        let data: Vec<f32> = (0..100 * 16).map(|i| ((i % 17) as f32) * 0.05).collect();
        let t: Tensor<Flex, 2> =
            Tensor::from_data(TensorData::new(data, [100, 16]), &Default::default());
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_softmax_f64() {
        // Exercises the softmax_last_dtype! + softmax_row_native f64 path.
        // Cross-check against burn_tensor::activation::softmax on f64.
        use burn_tensor::{Tensor, TensorData, TensorPrimitive};
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(data.to_vec(), [2, 4]),
            (&Default::default(), burn_backend::DType::F64),
        );
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f64>(&reference.into_data(), Tolerance::absolute(1e-10));
    }

    #[test]
    fn test_softmax_f16() {
        // Exercises the softmax_last_dtype! + softmax_row_half f16 path.
        // f16 has ~1e-3 precision (11-bit mantissa), so cross-check with a
        // matching tolerance against burn_tensor::activation::softmax.
        use burn_std::f16;
        use burn_tensor::{Tensor, TensorData, TensorPrimitive};
        let data: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(data, [2, 4]),
            (&Default::default(), burn_backend::DType::F16),
        );
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f16>(&reference.into_data(), Tolerance::absolute(1e-2));
    }

    #[test]
    fn test_softmax_bf16() {
        // Exercises the softmax_last_dtype! + softmax_row_half bf16 path.
        // bf16 has ~1e-2 precision (8-bit mantissa).
        use burn_std::bf16;
        use burn_tensor::{Tensor, TensorData, TensorPrimitive};
        let data: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(data, [2, 4]),
            (&Default::default(), burn_backend::DType::BF16),
        );
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<bf16>(&reference.into_data(), Tolerance::absolute(5e-2));
    }

    #[test]
    fn test_layer_norm_multi_chunk_rayon() {
        // 128 rows > ROWS_PER_TASK (64) so rayon produces multiple tasks.
        // d_model=16 also exercises the SIMD body across common lane widths
        // (NEON 4, AVX2 8, AVX-512 16).
        use burn_tensor::TensorPrimitive;
        let data: Vec<f32> = (0..128 * 16).map(|i| ((i % 19) as f32) * 0.03).collect();
        let t: Tensor<Flex, 2> =
            Tensor::from_data(TensorData::new(data, [128, 16]), &Default::default());
        let gamma: Tensor<Flex, 1> = Tensor::from_data([1.0f32; 16], &Default::default());
        let beta: Tensor<Flex, 1> = Tensor::from_data([0.0f32; 16], &Default::default());

        // Reference: manually compute per-row (x - mean) / sqrt(var + eps),
        // same formula the fused path implements.
        let rows_in = t.clone().into_data().to_vec::<f32>().unwrap();
        let mut expected = vec![0.0f32; rows_in.len()];
        let eps = 1e-5f32;
        for (in_row, out_row) in rows_in.chunks(16).zip(expected.chunks_mut(16)) {
            let mean: f32 = in_row.iter().sum::<f32>() / 16.0;
            let var: f32 = in_row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 16.0;
            let inv_std = 1.0 / (var + eps).sqrt();
            for (i, &x) in in_row.iter().enumerate() {
                out_row[i] = (x - mean) * inv_std;
            }
        }

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), 1e-5);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, [128, 16]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_non_contiguous_input() {
        // A transposed tensor has non-contiguous strides. softmax calls
        // `to_contiguous()` internally, and the fused kernel reads via
        // `.storage::<f32>()` which would read stale data if the
        // to_contiguous call were ever dropped. This test pins the contract.
        use burn_tensor::TensorPrimitive;
        // [3, 4] tensor, then transpose to [4, 3] (non-contiguous), softmax last axis.
        let t: Tensor<Flex, 2> = Tensor::from_data(
            [
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &Default::default(),
        );
        let t_transposed = t.transpose();
        let reference = activation::softmax(t_transposed.clone(), 1);

        let primitive = match t_transposed.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_softmax_empty_last_dim_returns_input() {
        // shape [2, 0] has zero elements; the early return in softmax_last_f32
        // is supposed to hand the input back unchanged rather than produce
        // NaN via 0/0. Locks that behavior so a future refactor that
        // removes the early-return check gets caught.
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(Vec::<f32>::new(), [2, 0]),
            &Default::default(),
        );
        let shape_before = t.shape();
        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let result = crate::ops::activation::softmax(primitive, 1);
        let result: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(result));
        assert_eq!(result.shape(), shape_before);
    }

    #[test]
    fn test_layer_norm_empty_last_dim_returns_input() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(Vec::<f32>::new(), [3, 0]),
            &Default::default(),
        );
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data(TensorData::new(Vec::<f32>::new(), [0]), &Default::default());
        let beta: Tensor<Flex, 1> =
            Tensor::from_data(TensorData::new(Vec::<f32>::new(), [0]), &Default::default());
        let shape_before = t.shape();

        let t_p = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_p = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_p = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let result = crate::ops::activation::layer_norm(t_p, g_p, Some(b_p), 1e-5);
        let result: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(result));
        assert_eq!(result.shape(), shape_before);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_length_mismatch_panics() {
        use burn_tensor::TensorPrimitive;
        // input last dim is 4, but gamma length is 3 — the assert_eq on
        // gamma.last == d_model should fire.
        let t: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], &Default::default());
        let gamma: Tensor<Flex, 1> = Tensor::from_data([1.0f32, 1.0, 1.0], &Default::default());

        let t_p = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_p = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let _ = crate::ops::activation::layer_norm(t_p, g_p, None, 1e-5);
    }

    #[test]
    #[should_panic(expected = "beta must be a 1-D tensor")]
    fn test_layer_norm_beta_length_mismatch_panics() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], &Default::default());
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data([1.0f32, 1.0, 1.0, 1.0], &Default::default());
        let beta: Tensor<Flex, 1> = Tensor::from_data([0.0f32, 0.0, 0.0], &Default::default());

        let t_p = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_p = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_p = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let _ = crate::ops::activation::layer_norm(t_p, g_p, Some(b_p), 1e-5);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_rank_mismatch_panics() {
        use burn_backend::DType;
        use burn_tensor::TensorPrimitive;
        // input last dim is 4, and gamma is [2, 4] — last dim matches d_model
        // but rank is 2. The old check (only last-dim == d_model) would have
        // accepted this and then indexed wrong data in the row kernel.
        let t: Tensor<Flex, 2> =
            Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], (&Default::default(), DType::F32));
        let gamma: Tensor<Flex, 2> = Tensor::from_data(
            [[1.0f32, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            (&Default::default(), DType::F32),
        );

        let t_p = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_p = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let _ = crate::ops::activation::layer_norm(t_p, g_p, None, 1e-5);
    }

    #[test]
    fn test_softmax_non_last_axis_matches_decomposed() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 3> = Tensor::from_data(
            [
                [[1.0f32, -2.0, 0.5], [3.0, 0.0, -1.0]],
                [[0.1, 2.5, -0.3], [1.2, -0.7, 2.1]],
            ],
            &Default::default(),
        );
        // Softmax on dim=1 (middle axis): fused path exercises the permute branch.
        let reference = burn_tensor::activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused_tensor: Tensor<Flex, 3> = Tensor::from_primitive(TensorPrimitive::Float(fused));
        fused_tensor
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), burn_tensor::Tolerance::default());
    }

    // Row length 17 is deliberately chosen: it exercises the "SIMD body ran N
    // elements, then scalar tail processes M > 0" combination on every common
    // SIMD width (NEON/SSE f32x4: body=16, tail=1; AVX2 f32x8: body=16, tail=1;
    // AVX-512 f32x16: body=16, tail=1). Row lengths that are exact multiples
    // of the SIMD width never hit the tail branch, so without a test like this
    // a bug in the scalar tail kernel would pass CI silently.
    #[test]
    fn test_softmax_simd_body_plus_scalar_tail() {
        use burn_backend::DType;
        use burn_tensor::TensorPrimitive;
        // 2 rows of length 17 with varied values so the max/exp/sum path is
        // meaningful per row.
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(data, [2, 17]),
            (&Default::default(), DType::F32),
        );
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_layer_norm_simd_body_plus_scalar_tail() {
        use burn_backend::DType;
        use burn_tensor::TensorPrimitive;
        // Same rationale as test_softmax_simd_body_plus_scalar_tail: row length
        // 17 leaves exactly one scalar-tail element after any common SIMD body.
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(data, [2, 17]),
            (&Default::default(), DType::F32),
        );
        let gamma_data: Vec<f32> = (0..17).map(|i| 1.0 + i as f32 * 0.05).collect();
        let beta_data: Vec<f32> = (0..17).map(|i| i as f32 * 0.01).collect();
        let gamma: Tensor<Flex, 1> = Tensor::from_data(
            TensorData::new(gamma_data, [17]),
            (&Default::default(), DType::F32),
        );
        let beta: Tensor<Flex, 1> = Tensor::from_data(
            TensorData::new(beta_data, [17]),
            (&Default::default(), DType::F32),
        );

        // Reference: manual layer_norm via primitive tensor ops.
        let mean = t.clone().mean_dim(1);
        let centered = t.clone() - mean;
        let var = centered.clone().powi_scalar(2).mean_dim(1);
        let eps = 1e-5f32;
        let normed = centered / (var + eps).sqrt();
        let reference = normed * gamma.clone().unsqueeze::<2>() + beta.clone().unsqueeze::<2>();

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), 1e-5);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_layer_norm_f64_with_beta_multi_chunk() {
        // 80 rows > ROWS_PER_TASK (64) exercises the rayon multi-chunk
        // f64 path and both pre- and post-welford arithmetic on f64.
        use burn_tensor::TensorPrimitive;
        let d_model = 16;
        let n_rows = 80;
        let data: Vec<f64> = (0..n_rows * d_model)
            .map(|i| ((i % 13) as f64) * 0.07 - 0.3)
            .collect();
        let dev_f64 = (&Default::default(), burn_backend::DType::F64);
        let t: Tensor<Flex, 2> =
            Tensor::from_data(TensorData::new(data, [n_rows, d_model]), dev_f64);
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data(TensorData::new(vec![0.9f64; d_model], [d_model]), dev_f64);
        let beta: Tensor<Flex, 1> =
            Tensor::from_data(TensorData::new(vec![0.05f64; d_model], [d_model]), dev_f64);

        let mean = t.clone().mean_dim(1);
        let centered = t.clone() - mean;
        let var = centered.clone().powi_scalar(2).mean_dim(1);
        let eps = 1e-5f64;
        let normed = centered / (var + eps).sqrt();
        let reference = normed * gamma.clone().unsqueeze::<2>() + beta.clone().unsqueeze::<2>();

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), eps);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f64>(&reference.into_data(), Tolerance::absolute(1e-10));
    }

    #[test]
    fn test_layer_norm_f64_no_beta() {
        use burn_tensor::TensorPrimitive;
        let dev_f64 = (&Default::default(), burn_backend::DType::F64);
        let t: Tensor<Flex, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0, -1.0, 0.5, 1.5, -0.5], [2, 4]),
            dev_f64,
        );
        let gamma: Tensor<Flex, 1> =
            Tensor::from_data(TensorData::new(vec![1.0f64; 4], [4]), dev_f64);

        let mean = t.clone().mean_dim(1);
        let centered = t.clone() - mean;
        let var = centered.clone().powi_scalar(2).mean_dim(1);
        let eps = 1e-5f64;
        let reference = centered / (var + eps).sqrt() * gamma.clone().unsqueeze::<2>();

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::layer_norm(t_prim, g_prim, None, eps);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused
            .into_data()
            .assert_approx_eq::<f64>(&reference.into_data(), Tolerance::absolute(1e-10));
    }

    // Shared body for f16/bf16 layer_norm tests. Parameterized on the
    // half-precision element type to avoid duplicating the scaffolding.
    fn check_layer_norm_half_precision<E>(from_f32: fn(f32) -> E, dtype: burn_backend::DType)
    where
        E: burn_tensor::Element + burn_backend::Element + num_traits::Float,
    {
        use burn_tensor::TensorPrimitive;
        let rows: [[f32; 4]; 3] = [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, 2.0],
            [0.5, -0.5, 1.5, -1.5],
        ];
        let dev_h = (&Default::default(), dtype);
        let data: Vec<E> = rows.iter().flatten().map(|&x| from_f32(x)).collect();
        let t: Tensor<Flex, 2> = Tensor::from_data(TensorData::new(data, [3, 4]), dev_h);
        let gamma_data: Vec<E> = [1.0f32, 0.5, 1.5, 1.0]
            .iter()
            .map(|&x| from_f32(x))
            .collect();
        let beta_data: Vec<E> = [0.1f32, -0.1, 0.0, 0.2]
            .iter()
            .map(|&x| from_f32(x))
            .collect();
        let gamma: Tensor<Flex, 1> = Tensor::from_data(TensorData::new(gamma_data, [4]), dev_h);
        let beta: Tensor<Flex, 1> = Tensor::from_data(TensorData::new(beta_data, [4]), dev_h);

        let mean = t.clone().mean_dim(1);
        let centered = t.clone() - mean;
        let var = centered.clone().powi_scalar(2).mean_dim(1);
        let eps = 1e-5f32;
        let normed = centered / (var + eps).sqrt();
        let reference = normed * gamma.clone().unsqueeze::<2>() + beta.clone().unsqueeze::<2>();

        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let b_prim = match beta.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::layer_norm(t_prim, g_prim, Some(b_prim), eps as f64);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        // Tolerance reflects the half-precision round-trip through the
        // f32 fused kernel. bf16 has ~1e-2 precision, f16 ~1e-3.
        fused
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(3e-2));
    }

    #[test]
    fn test_layer_norm_f16_via_f32_cast() {
        check_layer_norm_half_precision::<burn_std::f16>(
            burn_std::f16::from_f32,
            burn_backend::DType::F16,
        );
    }

    #[test]
    fn test_layer_norm_bf16_via_f32_cast() {
        check_layer_norm_half_precision::<burn_std::bf16>(
            burn_std::bf16::from_f32,
            burn_backend::DType::BF16,
        );
    }

    #[test]
    fn test_softmax_non_last_axis_rank4_dim0() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 4> = Tensor::from_data(
            [
                [[[1.0f32, -0.5], [0.3, 2.1]], [[-1.2, 0.0], [0.8, 1.5]]],
                [[[0.4, -1.1], [2.0, 0.2]], [[-0.3, 0.9], [1.1, -0.7]]],
            ],
            &Default::default(),
        );
        let reference = burn_tensor::activation::softmax(t.clone(), 0);

        let prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(prim, 0);
        let fused_tensor: Tensor<Flex, 4> = Tensor::from_primitive(TensorPrimitive::Float(fused));
        fused_tensor
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "softmax dim")]
    fn test_softmax_dim_out_of_range_panics() {
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> =
            Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &Default::default());
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let _ = crate::ops::activation::softmax(prim, 2);
    }

    #[test]
    #[should_panic(expected = "gamma dtype")]
    fn test_layer_norm_gamma_dtype_mismatch_panics() {
        // Input is f32 (default), gamma is explicitly f64. Verifies that
        // `layer_norm` rejects the mismatch up front rather than panicking
        // later inside the storage-typed access.
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0, 4.0]], &Default::default());
        let gamma: Tensor<Flex, 1> = Tensor::from_data(
            TensorData::new(vec![1.0f64; 4], [4]),
            (&Default::default(), burn_backend::DType::F64),
        );
        let t_prim = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let g_prim = match gamma.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let _ = crate::ops::activation::layer_norm(t_prim, g_prim, None, 1e-5);
    }

    #[test]
    fn test_log_sigmoid() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // log_sigmoid(-10) ~ -10, log_sigmoid(0) = ln(0.5) = -0.6931..., log_sigmoid(10) ~ 0
        activation::log_sigmoid(t)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-10.0, -0.6931472, 0.0]),
                Tolerance::absolute(1e-3),
            );
    }
}
