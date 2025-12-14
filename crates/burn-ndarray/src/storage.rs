//! Copy-on-write storage for zero-copy tensor loading.
//!
//! This module provides `NdArrayStorage<E>`, which enables true zero-copy loading
//! from burnpack files. When data is borrowed from external memory (like mmap'd files
//! or static data), it remains zero-copy until a mutating operation is performed,
//! at which point it's copied (copy-on-write semantics).
//!
//! This integrates with ndarray's existing COW patterns - operations that check
//! `is_unique()` will see borrowed data as non-unique, triggering the allocation path.

use alloc::vec::Vec;
use burn_backend::Element;
use burn_std::Bytes;
use core::mem;
use ndarray::{ArcArray, ArrayView, IxDyn};

/// Storage that supports both owned data and borrowed (zero-copy) data.
///
/// # Copy-on-Write Semantics
///
/// - **Borrowed**: Data from external source (burnpack, mmap, static).
///   Reports `is_unique() == false` to trigger copy on mutation.
/// - **Owned**: Standard `ArcArray` with built-in COW via Arc refcount.
///
/// # Example
///
/// ```ignore
/// // Zero-copy load
/// let storage = NdArrayStorage::from_borrowed(bytes, shape);
/// storage.is_unique();  // false - will copy on mutation
///
/// // Read operations use view() - zero-copy
/// let view = storage.view();
///
/// // Mutation converts to owned
/// let owned = storage.into_owned();  // Copies here
/// ```
#[derive(Debug)]
pub enum NdArrayStorage<E: Element> {
    /// Borrowed from external source (e.g., burnpack zero-copy load).
    /// Keeps `Bytes` alive to ensure the referenced memory is valid.
    Borrowed {
        /// Source bytes - keeps external memory alive via reference counting
        bytes: Bytes,
        /// Shape of the tensor
        shape: Vec<usize>,
    },

    /// Standard owned storage with ArcArray COW semantics.
    Owned(ArcArray<E, IxDyn>),
}

impl<E: Element> Clone for NdArrayStorage<E> {
    fn clone(&self) -> Self {
        match self {
            // For borrowed data, clone the Bytes (cheap Arc clone) and shape
            Self::Borrowed { bytes, shape } => Self::Borrowed {
                bytes: bytes.clone(),
                shape: shape.clone(),
            },
            // For owned data, clone the ArcArray (cheap Arc clone)
            Self::Owned(arr) => Self::Owned(arr.clone()),
        }
    }
}

impl<E: Element> NdArrayStorage<E> {
    /// Create borrowed storage from external bytes.
    ///
    /// Returns the bytes and shape back on failure (misaligned or too small),
    /// enabling zero-copy even for native allocations by avoiding defensive cloning.
    ///
    /// # Requirements
    ///
    /// The caller must ensure that:
    /// - The `Bytes` contain valid data for the element type `E`
    /// - The data is contiguous in row-major (C) order matching the provided shape
    ///
    /// These requirements are upheld when loading from `TensorData` (burnpack, etc.)
    /// which always stores data contiguously in row-major order.
    pub fn from_borrowed(bytes: Bytes, shape: Vec<usize>) -> Result<Self, (Bytes, Vec<usize>)> {
        // Validate alignment
        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(mem::align_of::<E>()) {
            return Err((bytes, shape));
        }

        // Validate size (using checked arithmetic to prevent overflow)
        let num_elements = match shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        {
            Some(n) => n,
            None => return Err((bytes, shape)),
        };
        let expected_size = match num_elements.checked_mul(mem::size_of::<E>()) {
            Some(s) => s,
            None => return Err((bytes, shape)),
        };
        if bytes.len() < expected_size {
            return Err((bytes, shape));
        }

        Ok(Self::Borrowed { bytes, shape })
    }

    /// Create owned storage from an ArcArray.
    #[inline]
    pub fn from_owned(array: ArcArray<E, IxDyn>) -> Self {
        Self::Owned(array)
    }

    /// Returns whether this storage is uniquely owned and can be mutated in-place.
    ///
    /// - **Borrowed**: Always returns `false` to trigger copy-on-write.
    /// - **Owned**: Delegates to `ArcArray::is_unique()`.
    ///
    /// This integrates with existing SIMD code patterns like:
    /// ```ignore
    /// if tensor.is_unique() {
    ///     // mutate in place
    /// } else {
    ///     // allocate new
    /// }
    /// ```
    #[inline]
    pub fn is_unique(&self) -> bool {
        match self {
            Self::Borrowed { .. } => false, // Force copy path
            Self::Owned(arr) => arr.is_unique(),
        }
    }

    /// Get a read-only view of the data.
    ///
    /// This is zero-copy for both borrowed and owned variants.
    #[inline]
    pub fn view(&self) -> ArrayView<'_, E, IxDyn> {
        match self {
            Self::Borrowed { bytes, shape } => {
                let ptr = bytes.as_ptr() as *const E;
                let dim = IxDyn(shape);
                // SAFETY:
                // - `bytes` is kept alive for the lifetime of `self`
                // - Alignment was validated in `from_borrowed`
                // - Size was validated in `from_borrowed`
                unsafe { ArrayView::from_shape_ptr(dim, ptr) }
            }
            Self::Owned(arr) => arr.view(),
        }
    }

    /// Convert to owned ArcArray.
    ///
    /// - **Borrowed**: Copies the data into a new ArcArray.
    /// - **Owned + unique**: Returns the array without copying.
    /// - **Owned + shared**: Clones the data.
    pub fn into_owned(self) -> ArcArray<E, IxDyn> {
        match self {
            Self::Borrowed { bytes, shape } => {
                let ptr = bytes.as_ptr() as *const E;
                let dim = IxDyn(&shape);
                // SAFETY: Same as view() - bytes is valid for this scope
                let view = unsafe { ArrayView::from_shape_ptr(dim, ptr) };
                view.to_owned().into_shared()
            }
            Self::Owned(arr) => arr,
        }
    }

    /// Convert to shared ArcArray, suitable for returning from operations.
    ///
    /// This is equivalent to `into_owned()` but named for clarity.
    #[inline]
    pub fn into_shared(self) -> ArcArray<E, IxDyn> {
        self.into_owned()
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Borrowed { shape, .. } => shape,
            Self::Owned(arr) => arr.shape(),
        }
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if this is borrowed (zero-copy) storage.
    #[inline]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed { .. })
    }

    /// Returns `true` if this is owned storage.
    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Ensure owned and return mutable reference to the ArcArray.
    ///
    /// Converts borrowed to owned if necessary.
    pub fn ensure_owned(&mut self) -> &mut ArcArray<E, IxDyn> {
        if let Self::Borrowed { bytes, shape } = self {
            let ptr = bytes.as_ptr() as *const E;
            let dim = IxDyn(shape);
            // SAFETY: Same as view()
            let view = unsafe { ArrayView::from_shape_ptr(dim, ptr) };
            *self = Self::Owned(view.to_owned().into_shared());
        }
        match self {
            Self::Owned(arr) => arr,
            Self::Borrowed { .. } => unreachable!(),
        }
    }
}

/// Convert from ArcArray to NdArrayStorage.
impl<E: Element> From<ArcArray<E, IxDyn>> for NdArrayStorage<E> {
    fn from(array: ArcArray<E, IxDyn>) -> Self {
        Self::Owned(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_std::Bytes;

    #[test]
    fn test_borrowed_is_not_unique() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        assert!(!storage.is_unique());
        assert!(storage.is_borrowed());
    }

    #[test]
    fn test_owned_unique_when_single_ref() {
        let array = ndarray::ArrayD::from_elem(IxDyn(&[2, 2]), 1.0f32).into_shared();
        let storage = NdArrayStorage::from_owned(array);

        assert!(storage.is_unique());
        assert!(storage.is_owned());
    }

    #[test]
    fn test_owned_not_unique_when_cloned() {
        let array = ndarray::ArrayD::from_elem(IxDyn(&[2, 2]), 1.0f32).into_shared();
        let storage = NdArrayStorage::from_owned(array);
        let _clone = storage.clone();

        assert!(!storage.is_unique());
    }

    #[test]
    fn test_view_zero_copy() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        let view = storage.view();
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[1, 1]], 4.0);
    }

    #[test]
    fn test_into_owned_copies_borrowed() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        let owned = storage.into_owned();
        assert_eq!(owned[[0, 0]], 1.0);
        assert_eq!(owned[[1, 1]], 4.0);
    }

    #[test]
    fn test_from_borrowed_validates_alignment() {
        use burn_std::AllocationProperty;

        // Test 1: Properly aligned data should succeed
        let aligned_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let aligned_bytes = Bytes::from_elems(aligned_data);

        // Verify test setup - should be 4-byte aligned for f32
        assert_eq!(
            (aligned_bytes.as_ptr() as usize) % core::mem::align_of::<f32>(),
            0,
            "Test setup: f32 data should be properly aligned"
        );

        let result = NdArrayStorage::<f32>::from_borrowed(aligned_bytes, vec![2, 2]);
        assert!(
            result.is_ok(),
            "from_borrowed should succeed for properly aligned data"
        );

        // Test 2: Misaligned data should fail
        // Create a buffer with 1 byte padding at start to force misalignment for f32
        let padded: &[u8] = &[
            0, // 1 byte padding to misalign
            0, 0, 128, 63, // 1.0f32 in little-endian
            0, 0, 0, 64, // 2.0f32
            0, 0, 64, 64, // 3.0f32
            0, 0, 128, 64, // 4.0f32
        ];
        let shared = bytes::Bytes::from_static(padded);
        // Slice starting at offset 1 to get misaligned pointer for f32
        let sliced = shared.slice(1..17); // 16 bytes = 4 f32s, but misaligned
        let misaligned_bytes = Bytes::from_shared(sliced, AllocationProperty::Other);

        // Verify test setup - should NOT be 4-byte aligned
        assert_ne!(
            (misaligned_bytes.as_ptr() as usize) % core::mem::align_of::<f32>(),
            0,
            "Test setup: sliced data should be misaligned for f32"
        );

        let result = NdArrayStorage::<f32>::from_borrowed(misaligned_bytes, vec![4]);
        assert!(
            result.is_err(),
            "from_borrowed should return Err for misaligned data"
        );
    }

    #[test]
    fn test_insufficient_size_returns_err() {
        // Create bytes that are too small for the requested shape
        let data: Vec<f32> = vec![1.0, 2.0]; // 8 bytes
        let bytes = Bytes::from_elems(data);

        // Try to create storage for 4 elements (needs 16 bytes)
        let result = NdArrayStorage::<f32>::from_borrowed(bytes, vec![4]);
        assert!(
            result.is_err(),
            "from_borrowed should return Err when bytes are too small"
        );
    }

    // ==========================================================================
    // Zero-copy hardening tests
    // These tests verify the zero-copy guarantee is maintained. If any of these
    // fail, it indicates a regression in zero-copy functionality.
    // ==========================================================================

    #[test]
    fn test_zero_copy_native_allocation() {
        // CRITICAL: Verify that native allocations (Bytes::from_elems) are zero-copy
        // on initial load. The view() must return a pointer to the SAME memory.
        //
        // Note: Native allocations copy on clone (this is expected), but the initial
        // load is still zero-copy, avoiding an extra copy in the common case where
        // the tensor is used without cloning.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let original_ptr = bytes.as_ptr();

        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        // Initial load must be zero-copy
        let view = storage.view();
        let view_ptr = view.as_ptr() as *const u8;

        assert_eq!(
            original_ptr, view_ptr,
            "ZERO-COPY REGRESSION: native allocation view() must return pointer to original bytes"
        );

        // Verify data integrity
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 1]], 2.0);
        assert_eq!(view[[1, 0]], 3.0);
        assert_eq!(view[[1, 1]], 4.0);
    }

    #[test]
    fn test_zero_copy_shared_bytes_pointer_identity() {
        // CRITICAL: Test with SharedBytesAllocationController for true zero-copy.
        // This simulates the actual burnpack/mmap loading path.
        use burn_std::AllocationProperty;

        // Create static-like data using bytes::Bytes
        let data: &[u8] = &[
            0, 0, 128, 63, // 1.0f32 in little-endian
            0, 0, 0, 64, // 2.0f32
            0, 0, 64, 64, // 3.0f32
            0, 0, 128, 64, // 4.0f32
        ];
        let shared = bytes::Bytes::from_static(data);
        let original_ptr = shared.as_ptr();

        // Create Bytes with SharedBytesAllocationController
        let bytes = Bytes::from_shared(shared, AllocationProperty::Other);

        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        // Verify pointer identity
        let view_ptr = storage.view().as_ptr() as *const u8;
        assert_eq!(
            original_ptr, view_ptr,
            "ZERO-COPY REGRESSION: SharedBytes view must point to original static data"
        );

        // Clone should also share the same memory
        let cloned = storage.clone();
        let cloned_ptr = cloned.view().as_ptr() as *const u8;
        assert_eq!(
            original_ptr, cloned_ptr,
            "ZERO-COPY REGRESSION: SharedBytes clone must share memory"
        );
    }

    #[test]
    fn test_clone_borrowed_stays_borrowed() {
        // Verify that cloning borrowed storage produces another borrowed storage.
        // Note: The underlying Bytes may or may not share memory depending on
        // the allocation controller (native allocations copy, file-backed may share).
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);

        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");
        let cloned = storage.clone();

        // Both should still be borrowed (the storage type is preserved)
        assert!(
            storage.is_borrowed(),
            "ZERO-COPY REGRESSION: original should remain borrowed after clone"
        );
        assert!(
            cloned.is_borrowed(),
            "ZERO-COPY REGRESSION: clone should be borrowed type"
        );

        // Both should report not unique (important for COW behavior)
        assert!(
            !storage.is_unique(),
            "ZERO-COPY REGRESSION: original should not be unique after clone"
        );
        assert!(
            !cloned.is_unique(),
            "ZERO-COPY REGRESSION: clone should not be unique"
        );

        // Data should be identical
        assert_eq!(storage.view(), cloned.view(), "Clone should have same data");
    }

    #[test]
    fn test_zero_copy_triggers_copy_on_mutation() {
        // Verify that into_owned() on borrowed data creates a NEW allocation
        // (this is the "copy" in copy-on-write)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let original_ptr = bytes.as_ptr();

        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        assert!(storage.is_borrowed(), "should start as borrowed");

        let owned = storage.into_owned();
        let owned_ptr = owned.as_ptr() as *const u8;

        assert_ne!(
            original_ptr, owned_ptr,
            "into_owned() on borrowed data MUST allocate new memory (copy-on-write)"
        );
    }

    #[test]
    fn test_borrowed_reports_not_unique() {
        // CRITICAL: Borrowed storage must report is_unique() == false
        // This is what triggers copy-on-write in mutation operations
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = Bytes::from_elems(data);
        let storage =
            NdArrayStorage::<f32>::from_borrowed(bytes, vec![2, 2]).expect("should create");

        assert!(
            !storage.is_unique(),
            "ZERO-COPY REGRESSION: borrowed storage MUST report is_unique() == false \
             to trigger copy-on-write. If this is true, mutations will corrupt shared data!"
        );
    }
}
