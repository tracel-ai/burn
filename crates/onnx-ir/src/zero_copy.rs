//! Zero-copy allocation controller for tensor loading
//!
//! This module provides `ZeroCopyAllocationController` which implements cubecl's
//! `AllocationController` trait, enabling zero-copy tensor data access from
//! `bytes::Bytes` buffers (which may be backed by memory-mapped files).

use burn_tensor::Bytes;
use cubecl_common::bytes::{AllocationController, AllocationError, AllocationProperty, SplitError};
use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// Allocation controller backed by `bytes::Bytes` for zero-copy access
///
/// This enables zero-copy tensor loading by referencing data directly from
/// the source buffer without copying into heap-allocated memory.
///
/// When used with memory-mapped files, the `bytes::Bytes` should be created via
/// `bytes::Bytes::from_owner(mmap)` which keeps the mmap alive through Arc.
///
/// # Safety
///
/// The underlying `bytes::Bytes` must remain valid for the lifetime of this
/// controller. This is ensured by `bytes::Bytes` using reference counting internally.
#[derive(Clone)]
pub struct ZeroCopyAllocationController {
    /// The backing bytes buffer (may contain mmap internally via from_owner)
    bytes: bytes::Bytes,
    /// Offset within the buffer where this allocation starts
    offset: usize,
    /// Length of this allocation in bytes
    len: usize,
}

impl ZeroCopyAllocationController {
    /// Create a new controller from a `bytes::Bytes` buffer
    ///
    /// The controller will reference a subslice of the buffer from
    /// `offset` to `offset + len`.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > bytes.len()`
    pub fn from_bytes(bytes: bytes::Bytes, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= bytes.len(),
            "ZeroCopyAllocationController: offset {} + len {} exceeds buffer length {}",
            offset,
            len,
            bytes.len()
        );
        Self { bytes, offset, len }
    }

    /// Create a new controller from a `bytes::Bytes` buffer using its full range
    pub fn from_bytes_full(bytes: bytes::Bytes) -> Self {
        let len = bytes.len();
        Self::from_bytes(bytes, 0, len)
    }

    /// Get a reference to the underlying byte slice
    fn as_slice(&self) -> &[u8] {
        &self.bytes[self.offset..self.offset + self.len]
    }

    /// Convert this controller into a cubecl `Bytes` type
    ///
    /// # Safety
    ///
    /// The caller must ensure that `self.len` accurately reflects the number
    /// of initialized bytes in the allocation.
    pub unsafe fn into_bytes(self) -> Bytes {
        let len = self.len;
        // SAFETY: caller guarantees len is the exact number of initialized bytes
        unsafe { Bytes::from_controller(Box::new(self), len) }
    }
}

impl AllocationController for ZeroCopyAllocationController {
    fn alloc_align(&self) -> usize {
        // Report byte alignment since we support arbitrary offsets within the buffer
        1
    }

    fn property(&self) -> AllocationProperty {
        // bytes::Bytes could be backed by mmap (via from_owner) or heap
        // We report Other since we don't know the underlying storage
        AllocationProperty::Other
    }

    fn memory(&self) -> &[MaybeUninit<u8>] {
        let slice = self.as_slice();
        // SAFETY: &[u8] and &[MaybeUninit<u8>] have the same memory layout,
        // and all bytes in the slice are initialized (coming from bytes::Bytes)
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const MaybeUninit<u8>, slice.len()) }
    }

    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // bytes::Bytes is conceptually read-only, but cubecl's Bytes type requires
        // memory_mut() to return the same-sized slice as memory() for DerefMut.
        // We return a "mutable" view but the caller should not actually write to it.
        //
        // This is a workaround for https://github.com/tracel-ai/cubecl/issues/1081
        // where `Bytes::try_into_vec()` unnecessarily requires mutable access for validation.
        //
        // SAFETY: The slice points to valid initialized memory. While bytes::Bytes
        // is logically immutable, providing a mutable reference is safe as long as
        // no actual mutation occurs (which cubecl's serialization doesn't do).
        let slice = self.as_slice();
        unsafe {
            std::slice::from_raw_parts_mut(slice.as_ptr() as *mut MaybeUninit<u8>, slice.len())
        }
    }

    fn split(
        &mut self,
        _offset: usize,
    ) -> Result<(Box<dyn AllocationController>, Box<dyn AllocationController>), SplitError> {
        Err(SplitError::Unsupported)
    }

    fn duplicate(&self) -> Option<Box<dyn AllocationController>> {
        // We can duplicate by cloning (bytes::Bytes handles ref counting)
        Some(Box::new(self.clone()))
    }

    unsafe fn copy_into(&self, buf: &mut [u8]) {
        let src = self.as_slice();
        let copy_len = buf.len().min(src.len());
        buf[..copy_len].copy_from_slice(&src[..copy_len]);
    }

    fn grow(&mut self, _size: usize, _align: usize) -> Result<(), AllocationError> {
        Err(AllocationError::UnsupportedOperation)
    }

    fn try_detach(&mut self) -> Option<NonNull<u8>> {
        // Memory is not managed by Rust's allocator
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bytes_full() {
        let data = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let controller = ZeroCopyAllocationController::from_bytes_full(data);

        assert_eq!(controller.as_slice(), &[1, 2, 3, 4, 5]);
        assert_eq!(controller.memory().len(), 5);
    }

    #[test]
    fn test_from_bytes_with_offset() {
        let data = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let controller = ZeroCopyAllocationController::from_bytes(data, 1, 3);

        assert_eq!(controller.as_slice(), &[2, 3, 4]);
        assert_eq!(controller.memory().len(), 3);
    }

    #[test]
    fn test_duplicate() {
        let data = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = ZeroCopyAllocationController::from_bytes_full(data);

        let dup = controller.duplicate().expect("duplicate should succeed");
        assert_eq!(dup.memory().len(), 3);
    }

    #[test]
    fn test_copy_into() {
        let data = bytes::Bytes::from_static(&[1, 2, 3, 4, 5]);
        let controller = ZeroCopyAllocationController::from_bytes_full(data);

        let mut buf = [0u8; 3];
        unsafe { controller.copy_into(&mut buf) };
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test]
    fn test_property() {
        let data = bytes::Bytes::from_static(&[1, 2, 3]);
        let controller = ZeroCopyAllocationController::from_bytes_full(data);

        assert!(matches!(controller.property(), AllocationProperty::Other));
    }

    #[test]
    #[should_panic(expected = "exceeds buffer length")]
    fn test_from_bytes_out_of_bounds() {
        let data = bytes::Bytes::from_static(&[1, 2, 3]);
        let _ = ZeroCopyAllocationController::from_bytes(data, 2, 5);
    }
}
