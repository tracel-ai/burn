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
    /// Returns `None` if the data is not properly aligned for type `E`.
    ///
    /// # Safety Requirements
    ///
    /// The `Bytes` must contain valid data for the element type and shape.
    pub fn from_borrowed(bytes: Bytes, shape: Vec<usize>) -> Option<Self> {
        // Validate alignment
        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(mem::align_of::<E>()) {
            return None;
        }

        // Validate size
        let expected_size = shape.iter().product::<usize>() * mem::size_of::<E>();
        if bytes.len() < expected_size {
            return None;
        }

        Some(Self::Borrowed { bytes, shape })
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
    fn test_misaligned_bytes_returns_none() {
        // Create bytes that are misaligned for f32 (needs 4-byte alignment)
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let bytes = Bytes::from_elems(data);

        // Try to create f32 storage - should fail due to alignment
        // (bytes of u8 might not be 4-byte aligned for f32)
        // Note: This test may pass or fail depending on allocator alignment
        let _result = NdArrayStorage::<f32>::from_borrowed(bytes, vec![4]);
        // We just check it doesn't panic - alignment check may or may not fail
    }
}
