//! Backend operations implementations.

use alloc::borrow::Cow;
use burn_backend::DType;
use burn_std::{bf16, f16};

use crate::FlexTensor;

/// The `DType` that matches `isize` on the current platform.
#[cfg(target_pointer_width = "64")]
pub(crate) const INDEX_DTYPE: DType = DType::I64;
/// The `DType` that matches `isize` on the current platform.
#[cfg(target_pointer_width = "32")]
pub(crate) const INDEX_DTYPE: DType = DType::I32;

/// Minimum total element count for rayon fan-out on per-row or per-chunk loops.
/// Below this, task-dispatch overhead dominates the per-unit work.
#[cfg(feature = "rayon")]
pub(crate) const PARALLEL_THRESHOLD: usize = 256 * 1024;

/// Wrapper for raw mutable pointers that can be sent across rayon threads.
///
/// # Safety
///
/// The caller must ensure:
/// - The pointer remains valid for the lifetime of all uses
/// - No two threads write to the same offset
/// - No references to the underlying data exist during writes
#[cfg(feature = "rayon")]
pub(crate) struct SendMutPtr<T>(*mut T);

#[cfg(feature = "rayon")]
unsafe impl<T> Send for SendMutPtr<T> {}
#[cfg(feature = "rayon")]
unsafe impl<T> Sync for SendMutPtr<T> {}

#[cfg(feature = "rayon")]
impl<T> SendMutPtr<T> {
    pub(crate) fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Write `val` at the given element offset.
    ///
    /// # Safety
    /// Offset must be in bounds and no other thread may write to the same offset.
    pub(crate) unsafe fn write(&self, offset: usize, val: T) {
        unsafe { self.0.add(offset).write(val) }
    }

    /// Returns the raw pointer offset by `offset` elements.
    ///
    /// # Safety
    /// Offset must be in bounds.
    pub(crate) unsafe fn ptr_add(&self, offset: usize) -> *mut T {
        unsafe { self.0.add(offset) }
    }
}

/// Read a float tensor's storage as f32 values, regardless of source dtype.
/// Returns a borrowed slice for F32 (zero-copy) and an owned Vec for other
/// float dtypes (F64, F16, BF16).
///
/// Returns elements in underlying buffer order. Callers needing logical
/// iteration order must call `to_contiguous()` first.
///
/// # Panics
/// Panics if the tensor's dtype is not one of F32, F64, F16, or BF16.
pub(crate) fn float_storage_as_f32(tensor: &FlexTensor) -> Cow<'_, [f32]> {
    match tensor.dtype() {
        DType::F32 => Cow::Borrowed(tensor.storage::<f32>()),
        DType::F64 => Cow::Owned(tensor.storage::<f64>().iter().map(|&x| x as f32).collect()),
        DType::F16 => Cow::Owned(
            tensor
                .storage::<f16>()
                .iter()
                .map(|x| f32::from(*x))
                .collect(),
        ),
        DType::BF16 => Cow::Owned(
            tensor
                .storage::<bf16>()
                .iter()
                .map(|x| f32::from(*x))
                .collect(),
        ),
        other => panic!("float_storage_as_f32: unsupported dtype {:?}", other),
    }
}

pub mod activation;
pub mod attention;
pub mod binary;
mod bool;
pub mod cat;
pub mod comparison;
#[macro_use]
mod conv_common;
pub mod conv;
pub mod conv_transpose;
pub mod cumulative;
pub mod deform_conv;
pub mod expand;
pub mod fft;
pub mod flip;
mod float;
pub mod gather_scatter;
pub mod grid_sample;
mod int;
pub mod interpolate;
pub mod mask;
pub mod matmul;
mod module;
pub mod pool;
mod qtensor;
pub mod reduce;
pub mod repeat_dim;
pub mod slice;
pub mod sort;
mod transaction;
pub mod unary;
pub mod unfold;
