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

// Tests kept here exercise flex-specific behavior: SIMD boundaries, rayon
// chunk boundaries, non-contiguous input handling, the flex-internal
// layer_norm op (no public API yet), and dtype-specific fused softmax
// paths (f16/bf16/f64). Plain activation/softmax smoke tests have been
// migrated to burn-backend-tests so they cover every backend. When adding
// new tests, keep them here only if they probe flex internals; otherwise
// add them to crates/burn-backend-tests/tests/tensor/float/activation/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::{DType, TensorData, TensorMetadata, Tolerance};
    use burn_std::{bf16, f16};
    use num_traits::Float;

    use crate::FlexTensor;

    // ============================================================================
    // Reference implementations (per-row, last-axis).
    //
    // These mirror the contract the fused kernel commits to: stable softmax via
    // (x - max), layer_norm via (x - mean) * inv(sqrt(var + eps)) with optional
    // affine. Written in plain Rust over f32/f64 slices so the tests avoid any
    // tensor-library dependency.
    // ============================================================================

    fn softmax_row<T: Float>(row_in: &[T], row_out: &mut [T]) {
        let max = row_in
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
        let mut sum = T::zero();
        for (i, &x) in row_in.iter().enumerate() {
            let e = (x - max).exp();
            row_out[i] = e;
            sum = sum + e;
        }
        for v in row_out.iter_mut() {
            *v = *v / sum;
        }
    }

    fn softmax_last_ref<T: Float>(data: &[T], row_len: usize) -> Vec<T> {
        let mut out = vec![T::zero(); data.len()];
        for (i, o) in data.chunks(row_len).zip(out.chunks_mut(row_len)) {
            softmax_row(i, o);
        }
        out
    }

    fn layer_norm_row<T: Float>(
        row_in: &[T],
        gamma: &[T],
        beta: Option<&[T]>,
        eps: T,
        row_out: &mut [T],
    ) {
        let n = T::from(row_in.len()).unwrap();
        let mean = row_in.iter().copied().fold(T::zero(), |a, b| a + b) / n;
        let var = row_in
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |a, b| a + b)
            / n;
        let inv_std = T::one() / (var + eps).sqrt();
        for (i, &x) in row_in.iter().enumerate() {
            let normed = (x - mean) * inv_std;
            let scaled = normed * gamma[i];
            row_out[i] = match beta {
                Some(b) => scaled + b[i],
                None => scaled,
            };
        }
    }

    fn layer_norm_last_ref<T: Float>(
        data: &[T],
        gamma: &[T],
        beta: Option<&[T]>,
        eps: T,
        row_len: usize,
    ) -> Vec<T> {
        let mut out = vec![T::zero(); data.len()];
        for (i, o) in data.chunks(row_len).zip(out.chunks_mut(row_len)) {
            layer_norm_row(i, gamma, beta, eps, o);
        }
        out
    }

    // ============================================================================
    // Helpers: FlexTensor constructors for typed inputs.
    // ============================================================================

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn flex_f64(data: Vec<f64>, shape: &[usize]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn flex_half<T: burn_backend::Element>(data: Vec<T>, shape: &[usize]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    // ============================================================================
    // layer_norm tests
    // ============================================================================

    #[test]
    fn test_layer_norm_2d_with_beta() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let gamma = flex_f32(vec![1.0; 4], &[4]);
        let beta = flex_f32(vec![0.0; 4], &[4]);
        let out = crate::ops::activation::layer_norm(t, gamma, Some(beta), 1e-5);

        let expected: Vec<f32> = vec![
            -1.3416408, -0.4472136, 0.4472136, 1.3416408, -1.3416408, -0.4472136, 0.4472136,
            1.3416408,
        ];
        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![2.0, 0.5, 1.0, 3.0], &[4]);
        let beta = flex_f32(vec![1.0, -1.0, 0.0, 2.0], &[4]);
        let out = crate::ops::activation::layer_norm(t, gamma, Some(beta), 1e-5);

        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // affine: [-1.6833, -1.2236, 0.4472, 6.0249]
        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![-1.6833, -1.2236, 0.4472, 6.0249], vec![1, 4]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_layer_norm_no_beta() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0; 4], &[4]);
        let out = crate::ops::activation::layer_norm(t, gamma, None, 1e-5);

        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                vec![-1.3416408, -0.4472136, 0.4472136, 1.3416408],
                vec![1, 4],
            ),
            Tolerance::absolute(1e-4),
        );
    }

    // ============================================================================
    // softmax SIMD / rayon boundary tests
    // ============================================================================

    #[test]
    fn test_softmax_simd_body_row() {
        // Row length 32 ensures the SIMD body runs on every supported target:
        // NEON (lanes=4), AVX2 (lanes=8), AVX-512 (lanes=16), SIMD128 (lanes=4).
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let expected = softmax_last_ref(&data, 32);
        let fused = crate::ops::activation::softmax(flex_f32(data, &[1, 32]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![1, 32]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_softmax_multi_chunk_rayon() {
        // 100 rows > ROWS_PER_TASK (64) triggers the rayon par_chunks path.
        let data: Vec<f32> = (0..100 * 16).map(|i| ((i % 17) as f32) * 0.05).collect();
        let expected = softmax_last_ref(&data, 16);
        let fused = crate::ops::activation::softmax(flex_f32(data, &[100, 16]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![100, 16]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_softmax_f64() {
        // Exercises softmax_last_dtype! + softmax_row_native f64 path.
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::ops::activation::softmax(flex_f64(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-10),
        );
    }

    #[test]
    fn test_softmax_f16() {
        // Exercises softmax_last_dtype! + softmax_row_half f16 path.
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let data: Vec<f16> = source.iter().map(|&x| f16::from_f32(x)).collect();
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::ops::activation::softmax(flex_half(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-2),
        );
    }

    #[test]
    fn test_softmax_bf16() {
        // Exercises softmax_last_dtype! + softmax_row_half bf16 path.
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let data: Vec<bf16> = source.iter().map(|&x| bf16::from_f32(x)).collect();
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::ops::activation::softmax(flex_half(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(5e-2),
        );
    }

    #[test]
    fn test_layer_norm_multi_chunk_rayon() {
        // 128 rows > ROWS_PER_TASK (64) triggers the rayon path.
        let data: Vec<f32> = (0..128 * 16).map(|i| ((i % 19) as f32) * 0.03).collect();
        let gamma_data: Vec<f32> = vec![1.0; 16];
        let beta_data: Vec<f32> = vec![0.0; 16];
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), 1e-5f32, 16);
        let fused = crate::ops::activation::layer_norm(
            flex_f32(data, &[128, 16]),
            flex_f32(gamma_data, &[16]),
            Some(flex_f32(beta_data, &[16])),
            1e-5,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![128, 16]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_empty_last_dim_returns_input() {
        // shape [2, 0]: empty last dim should round-trip unchanged instead
        // of producing NaN via 0/0.
        let t = flex_f32(Vec::<f32>::new(), &[2, 0]);
        let result = crate::ops::activation::softmax(t, 1);
        assert_eq!(result.shape().as_slice(), &[2, 0]);
    }

    #[test]
    fn test_layer_norm_empty_last_dim_returns_input() {
        let t = flex_f32(Vec::<f32>::new(), &[3, 0]);
        let gamma = flex_f32(Vec::<f32>::new(), &[0]);
        let beta = flex_f32(Vec::<f32>::new(), &[0]);
        let result = crate::ops::activation::layer_norm(t, gamma, Some(beta), 1e-5);
        assert_eq!(result.shape().as_slice(), &[3, 0]);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_length_mismatch_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0, 1.0, 1.0], &[3]);
        let _ = crate::ops::activation::layer_norm(t, gamma, None, 1e-5);
    }

    #[test]
    #[should_panic(expected = "beta must be a 1-D tensor")]
    fn test_layer_norm_beta_length_mismatch_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0, 1.0, 1.0, 1.0], &[4]);
        let beta = flex_f32(vec![0.0, 0.0, 0.0], &[3]);
        let _ = crate::ops::activation::layer_norm(t, gamma, Some(beta), 1e-5);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_rank_mismatch_panics() {
        // gamma [2, 4] has matching last-dim but rank 2, so the old last-dim
        // check alone would have accepted it and then indexed wrong storage.
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0; 8], &[2, 4]);
        let _ = crate::ops::activation::layer_norm(t, gamma, None, 1e-5);
    }

    // Row length 17 leaves exactly one scalar-tail element after every common
    // SIMD width (NEON/SSE f32x4: body=16, tail=1; AVX2 f32x8: body=16, tail=1;
    // AVX-512 f32x16: body=16, tail=1). Row lengths that divide evenly by the
    // SIMD width skip the tail branch entirely, so a bug in the scalar tail
    // kernel would sail past CI without a test like this.
    #[test]
    fn test_softmax_simd_body_plus_scalar_tail() {
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let expected = softmax_last_ref(&data, 17);
        let fused = crate::ops::activation::softmax(flex_f32(data, &[2, 17]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 17]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_layer_norm_simd_body_plus_scalar_tail() {
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let gamma_data: Vec<f32> = (0..17).map(|i| 1.0 + i as f32 * 0.05).collect();
        let beta_data: Vec<f32> = (0..17).map(|i| i as f32 * 0.01).collect();
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), 1e-5f32, 17);
        let fused = crate::ops::activation::layer_norm(
            flex_f32(data, &[2, 17]),
            flex_f32(gamma_data, &[17]),
            Some(flex_f32(beta_data, &[17])),
            1e-5,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 17]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_layer_norm_f64_with_beta_multi_chunk() {
        // 80 rows > ROWS_PER_TASK (64) exercises the rayon multi-chunk f64 path.
        let d_model = 16;
        let n_rows = 80;
        let data: Vec<f64> = (0..n_rows * d_model)
            .map(|i| ((i % 13) as f64) * 0.07 - 0.3)
            .collect();
        let gamma_data: Vec<f64> = vec![0.9; d_model];
        let beta_data: Vec<f64> = vec![0.05; d_model];
        let eps = 1e-5f64;
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), eps, d_model);
        let fused = crate::ops::activation::layer_norm(
            flex_f64(data, &[n_rows, d_model]),
            flex_f64(gamma_data, &[d_model]),
            Some(flex_f64(beta_data, &[d_model])),
            eps,
        );
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![n_rows, d_model]),
            Tolerance::absolute(1e-10),
        );
    }

    #[test]
    fn test_layer_norm_f64_no_beta() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 1.5, -0.5];
        let gamma_data: Vec<f64> = vec![1.0; 4];
        let eps = 1e-5f64;
        let expected = layer_norm_last_ref(&data, &gamma_data, None, eps, 4);
        let fused = crate::ops::activation::layer_norm(
            flex_f64(data, &[2, 4]),
            flex_f64(gamma_data, &[4]),
            None,
            eps,
        );
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-10),
        );
    }

    // Shared body for f16/bf16 layer_norm tests. The fused half-precision
    // kernel casts to f32 internally, so the reference is computed in f32
    // and compared back against the half output with an f32 tolerance.
    fn check_layer_norm_half_precision<E>(from_f32: fn(f32) -> E, dtype: DType)
    where
        E: burn_backend::Element + Float,
    {
        let rows_f32: [f32; 12] = [
            1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 1.5, -1.5,
        ];
        let gamma_f32: [f32; 4] = [1.0, 0.5, 1.5, 1.0];
        let beta_f32: [f32; 4] = [0.1, -0.1, 0.0, 0.2];
        let eps = 1e-5f32;

        let expected_f32 = layer_norm_last_ref(&rows_f32, &gamma_f32, Some(&beta_f32), eps, 4);

        let data: Vec<E> = rows_f32.iter().map(|&x| from_f32(x)).collect();
        let gamma_data: Vec<E> = gamma_f32.iter().map(|&x| from_f32(x)).collect();
        let beta_data: Vec<E> = beta_f32.iter().map(|&x| from_f32(x)).collect();
        assert_eq!(E::dtype(), dtype);

        let fused = crate::ops::activation::layer_norm(
            flex_half(data, &[3, 4]),
            flex_half(gamma_data, &[4]),
            Some(flex_half(beta_data, &[4])),
            eps as f64,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected_f32, vec![3, 4]),
            Tolerance::absolute(3e-2),
        );
    }

    #[test]
    fn test_layer_norm_f16_via_f32_cast() {
        check_layer_norm_half_precision::<f16>(f16::from_f32, DType::F16);
    }

    #[test]
    fn test_layer_norm_bf16_via_f32_cast() {
        check_layer_norm_half_precision::<bf16>(bf16::from_f32, DType::BF16);
    }

    #[test]
    #[should_panic(expected = "softmax dim")]
    fn test_softmax_dim_out_of_range_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = crate::ops::activation::softmax(t, 2);
    }

    #[test]
    #[should_panic(expected = "gamma dtype")]
    fn test_layer_norm_gamma_dtype_mismatch_panics() {
        // Input f32, gamma f64: layer_norm rejects the mismatch up front
        // rather than panicking later inside the storage-typed access.
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f64(vec![1.0; 4], &[4]);
        let _ = crate::ops::activation::layer_norm(t, gamma, None, 1e-5);
    }
}
