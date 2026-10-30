//! SIMD-optimized kernels for tensor operations.
//!
//! Provides portable SIMD implementations via `macerator` with automatic
//! dispatch to the best available instruction set:
//! - aarch64: NEON
//! - x86_64: AVX2, AVX512, SSE
//! - wasm32: SIMD128
//! - Other: Scalar fallback
//!
//! Enable with the `simd` feature flag (enabled by default).

// Portable SIMD kernels using macerator (reductions, scatter-add)
#[cfg(feature = "simd")]
pub mod kernels;

// SIMD-aligned memory allocation
#[cfg(feature = "simd")]
pub mod aligned;

// When simd feature enabled: use portable macerator for binary/comparison/bool ops
#[cfg(feature = "simd")]
mod portable;

#[cfg(feature = "simd")]
pub use portable::{
    CmpOp, abs_inplace_f32, add_inplace_f32, bool_and_inplace_u8, bool_and_u8, bool_not_inplace_u8,
    bool_not_u8, bool_or_inplace_u8, bool_or_u8, bool_xor_inplace_u8, bool_xor_u8, cmp_f32,
    cmp_scalar_f32, div_inplace_f32, mask_fill_f32, mask_fill_f64, mask_fill_i64, mask_fill_u8,
    mask_where_f32, mask_where_f64, mask_where_i64, mask_where_u8, mul_inplace_f32,
    rdiv_inplace_f32, recip_inplace_f32, rsub_inplace_f32, sub_inplace_f32,
};

#[cfg(feature = "simd")]
pub(crate) use portable::{
    Acos, Asin, Atan, Atan2, Atanh, Ceil, Cos, Cosh, Erf, Exp, Floor, Gelu, GeluBackward, GeluGrad,
    Hypot, LaneOp, LaneOp2, Ln, Ln1p, LogSigmoid, LogSigmoidBackward, LogSigmoidGrad, MAX_LANES,
    RoundTiesEven, Sigmoid, Sin, Sinh, Swap, Tan, Tanh, Trunc, exp_f32, map_inplace, map_to_vec,
    zip_map_inplace, zip_map_to_vec,
};

// When simd feature disabled: use scalar fallback (bool ops + CmpOp only)
#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(not(feature = "simd"))]
pub use scalar::{
    CmpOp, bool_and_inplace_u8, bool_and_u8, bool_not_inplace_u8, bool_not_u8, bool_or_inplace_u8,
    bool_or_u8, bool_xor_inplace_u8, bool_xor_u8,
};
