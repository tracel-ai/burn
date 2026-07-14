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
    CmpOp, abs_inplace_f32, add_inplace_f32, add_shared_row_inplace_f32, bool_and_inplace_u8,
    bool_and_u8, bool_not_inplace_u8, bool_not_u8, bool_or_inplace_u8, bool_or_u8,
    bool_xor_inplace_u8, bool_xor_u8, cmp_f32, cmp_scalar_f32, div_inplace_f32,
    div_shared_row_inplace_f32, mask_fill_f32, mask_fill_f64, mask_fill_i64, mask_fill_u8,
    mask_where_f32, mask_where_f64, mask_where_i64, mask_where_u8, mul_inplace_f32,
    mul_shared_row_inplace_f32, recip_inplace_f32, sub_inplace_f32, sub_shared_row_inplace_f32,
};

// When simd feature disabled: use scalar fallback (bool ops + CmpOp only)
#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(not(feature = "simd"))]
pub use scalar::{
    CmpOp, bool_and_inplace_u8, bool_and_u8, bool_not_inplace_u8, bool_not_u8, bool_or_inplace_u8,
    bool_or_u8, bool_xor_inplace_u8, bool_xor_u8,
};
