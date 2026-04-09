#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use burn_backend::{DType, Element, TensorData, TensorMetadata};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::layout::Layout;

/// CPU tensor primitive for the Flex backend.
///
/// Uses type-erased byte storage with runtime dtype and Arc-based sharing.
/// Clone is O(1) (refcount increment). Copy-on-write for mutations.
#[derive(Clone)]
pub struct FlexTensor {
    /// Shared byte storage. Clone increments refcount.
    data: Arc<Bytes>,
    /// Layout describing shape, strides, and offset.
    layout: Layout,
    /// Runtime data type.
    dtype: DType,
}

impl fmt::Debug for FlexTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlexTensor")
            .field("shape", self.layout.shape())
            .field("dtype", &self.dtype)
            .field("contiguous", &self.layout.is_contiguous())
            .field("unique", &self.is_unique())
            .finish()
    }
}

impl FlexTensor {
    /// Create a new tensor from bytes, layout, and dtype.
    pub fn new(data: Bytes, layout: Layout, dtype: DType) -> Self {
        Self {
            data: Arc::new(data),
            layout,
            dtype,
        }
    }

    /// Create a tensor from TensorData.
    pub fn from_data(data: TensorData) -> Self {
        let shape = data.shape.clone();
        let layout = Layout::contiguous(shape);
        let dtype = data.dtype;
        Self {
            data: Arc::new(data.bytes),
            layout,
            dtype,
        }
    }

    /// Convert tensor to TensorData.
    ///
    /// If non-contiguous or shared, this will copy data.
    pub fn into_data(self) -> TensorData {
        if self.layout.is_contiguous() && self.layout.start_offset() == 0 {
            let expected_bytes = self.layout.num_elements() * dtype_size(self.dtype);
            assert!(
                expected_bytes <= self.data.len(),
                "into_data: buffer ({} bytes) too small for {} elements of {:?}",
                self.data.len(),
                self.layout.num_elements(),
                self.dtype
            );
            if self.data.len() == expected_bytes {
                // Buffer exactly matches logical size; try zero-copy unwrap
                match Arc::try_unwrap(self.data) {
                    Ok(bytes) => TensorData {
                        bytes,
                        shape: self.layout.shape().clone(),
                        dtype: self.dtype,
                    },
                    Err(arc) => {
                        let bytes = Bytes::from_bytes_vec((*arc)[..expected_bytes].to_vec());
                        TensorData {
                            bytes,
                            shape: self.layout.shape().clone(),
                            dtype: self.dtype,
                        }
                    }
                }
            } else {
                // Contiguous at offset 0 but buffer is oversized (e.g., narrowed view).
                // Truncate to exact logical size.
                let bytes = Bytes::from_bytes_vec(self.data[..expected_bytes].to_vec());
                TensorData {
                    bytes,
                    shape: self.layout.shape().clone(),
                    dtype: self.dtype,
                }
            }
        } else {
            // Non-contiguous or non-zero offset: copy to contiguous layout
            self.to_contiguous().into_data()
        }
    }

    /// Check if this tensor has exclusive ownership of its data.
    ///
    /// When true, in-place mutations are safe without copying.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Get the layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Create a new tensor with a different layout but sharing the same data.
    ///
    /// This is a zero-copy operation used for operations like flip, transpose, etc.
    pub fn with_layout(self, layout: Layout) -> Self {
        Self {
            data: self.data,
            layout,
            dtype: self.dtype,
        }
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Check if tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Get the raw bytes (read-only).
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get a clone of the Arc for sharing data with a new layout.
    ///
    /// Use this for zero-copy view operations (reshape, transpose, slice).
    pub fn data_arc(&self) -> Arc<Bytes> {
        Arc::clone(&self.data)
    }

    /// Create a tensor from shared data, layout, and dtype.
    ///
    /// Use this for zero-copy view operations.
    pub fn from_arc(data: Arc<Bytes>, layout: Layout, dtype: DType) -> Self {
        Self {
            data,
            layout,
            dtype,
        }
    }

    /// Zero-copy typed view of the full storage buffer.
    ///
    /// Use with `StridedIter` for non-contiguous access, or with
    /// `layout().contiguous_offsets()` for the contiguous fast path.
    ///
    /// # Panics
    /// Panics if `E::dtype()` doesn't match the tensor's dtype.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn storage<E: Element + bytemuck::Pod>(&self) -> &[E] {
        assert!(
            E::dtype() == self.dtype
                || (matches!(
                    self.dtype,
                    DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8)
                ) && E::dtype() == DType::U8),
            "storage: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        bytemuck::cast_slice(&self.data)
    }

    /// Mutable typed view with copy-on-write semantics.
    ///
    /// If the tensor is shared (refcount > 1), this will copy the data first.
    /// For in-place operations, prefer `try_storage_mut()` which returns None
    /// if shared, allowing you to choose an alternative strategy.
    ///
    /// # Panics
    /// Panics if `E::dtype()` doesn't match the tensor's dtype.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn storage_mut<E: Element + bytemuck::Pod>(&mut self) -> &mut [E] {
        assert!(
            E::dtype() == self.dtype
                || (matches!(
                    self.dtype,
                    DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8)
                ) && E::dtype() == DType::U8),
            "storage_mut: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        // COW: clone data if shared
        let bytes = Arc::make_mut(&mut self.data);
        bytemuck::cast_slice_mut(bytes)
    }

    /// Try to get mutable storage without copying.
    ///
    /// Returns `Some` if tensor is uniquely owned, `None` if shared.
    /// Use this when you want to avoid the implicit copy in `storage_mut()`.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn try_storage_mut<E: Element + bytemuck::Pod>(&mut self) -> Option<&mut [E]> {
        assert!(
            E::dtype() == self.dtype
                || (matches!(
                    self.dtype,
                    DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8)
                ) && E::dtype() == DType::U8),
            "try_storage_mut: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        if self.is_unique() {
            // Safe: we're the only owner
            let bytes = Arc::get_mut(&mut self.data)?;
            Some(bytemuck::cast_slice_mut(bytes))
        } else {
            None
        }
    }

    /// Get typed slice view (zero-cost if contiguous and offset is 0).
    ///
    /// Returns None if dtype doesn't match E or tensor is non-contiguous.
    pub fn as_slice<E: Element + bytemuck::Pod>(&self) -> Option<&[E]> {
        if E::dtype() != self.dtype {
            return None;
        }
        let storage: &[E] = self.storage();
        self.layout
            .contiguous_offsets()
            .map(|(start, end)| &storage[start..end])
    }

    /// Create an empty tensor with given shape and dtype.
    pub fn empty(shape: Shape, dtype: DType) -> Self {
        let num_elements = shape.num_elements();
        let elem_size = dtype_size(dtype);
        let bytes = Bytes::from_bytes_vec(alloc::vec![0u8; num_elements * elem_size]);
        let layout = Layout::contiguous(shape);
        Self {
            data: Arc::new(bytes),
            layout,
            dtype,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        Self::empty(shape, dtype)
    }

    /// Create a tensor filled with `n` copies of a typed value.
    pub fn filled_typed<E: bytemuck::Pod + Send + Sync>(
        shape: Shape,
        dtype: DType,
        value: E,
    ) -> Self {
        assert_eq!(
            dtype_size(dtype),
            core::mem::size_of::<E>(),
            "filled_typed: dtype size mismatch"
        );
        let n = shape.num_elements();
        let data = alloc::vec![value; n];
        let bytes = Bytes::from_elems(data);
        Self {
            data: Arc::new(bytes),
            layout: Layout::contiguous(shape),
            dtype,
        }
    }

    /// Copy to contiguous layout if needed.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() && self.layout.start_offset() == 0 {
            return self.clone();
        }

        // Copy data to new contiguous buffer
        match self.dtype {
            DType::F64 => self.copy_contiguous::<f64>(),
            DType::F32 => self.copy_contiguous::<f32>(),
            DType::F16 => self.copy_contiguous::<f16>(),
            DType::BF16 => self.copy_contiguous::<bf16>(),
            DType::I64 => self.copy_contiguous::<i64>(),
            DType::I32 => self.copy_contiguous::<i32>(),
            DType::I16 => self.copy_contiguous::<i16>(),
            DType::I8 => self.copy_contiguous::<i8>(),
            DType::U64 => self.copy_contiguous::<u64>(),
            DType::U32 => self.copy_contiguous::<u32>(),
            DType::U16 => self.copy_contiguous::<u16>(),
            DType::U8 => self.copy_contiguous::<u8>(),
            DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8) => {
                self.copy_contiguous::<u8>()
            }
            DType::Bool(burn_std::BoolStore::U32) => {
                panic!("burn-flex: Bool(U32) storage is not yet supported")
            }
            _ => panic!("Unsupported dtype for contiguous copy: {:?}", self.dtype),
        }
    }

    fn copy_contiguous<E: Element + bytemuck::Pod>(&self) -> Self {
        let src: &[E] = bytemuck::cast_slice(&self.data);
        let n = self.layout.num_elements();
        let mut dst = Vec::with_capacity(n);

        // Fast path for 2D positive-stride tensors (common for transpose).
        // Tiles both dimensions so source reads stay within a bounded memory
        // region per tile, reducing L2/L3 cache pressure for transposed matrices
        // where one stride is large.
        if let Some((rows, cols, row_stride, col_stride)) = self.layout.as_2d_strides()
            && row_stride >= 0
            && col_stride >= 0
        {
            const TILE: usize = 16;
            let offset = self.layout.start_offset() as isize;

            // SAFETY: capacity is n. The tiled loops visit every (row, col)
            // pair exactly once, writing all n positions. Indices are bounded:
            // dst_base + c = row * cols + col_tile + c < rows * cols == n.
            debug_assert_eq!(rows * cols, n, "2D strides must cover all elements");
            unsafe { dst.set_len(n) };

            for row_tile in (0..rows).step_by(TILE) {
                let row_end = (row_tile + TILE).min(rows);
                for col_tile in (0..cols).step_by(TILE) {
                    let col_end = (col_tile + TILE).min(cols);
                    for row in row_tile..row_end {
                        let row_base =
                            offset + row as isize * row_stride + col_tile as isize * col_stride;
                        let dst_base = row * cols + col_tile;
                        for c in 0..(col_end - col_tile) {
                            let idx = (row_base + c as isize * col_stride) as usize;
                            unsafe {
                                *dst.get_unchecked_mut(dst_base + c) = src[idx];
                            }
                        }
                    }
                }
            }
        } else {
            // General fallback using strided iterator
            for idx in crate::strided_index::StridedIter::new(&self.layout) {
                dst.push(src[idx]);
            }
        }

        let bytes = Bytes::from_elems(dst);
        let layout = Layout::contiguous(self.layout.shape().clone());
        Self {
            data: Arc::new(bytes),
            layout,
            dtype: self.dtype,
        }
    }

    /// Reshape tensor. Zero-copy if contiguous.
    pub fn reshape(&self, new_shape: Shape) -> Self {
        assert_eq!(
            self.layout.num_elements(),
            new_shape.num_elements(),
            "reshape must preserve total elements"
        );

        if let Some(new_layout) = self.layout.reshape(new_shape.clone()) {
            Self {
                data: Arc::clone(&self.data),
                layout: new_layout,
                dtype: self.dtype,
            }
        } else {
            // Non-contiguous: copy first
            self.to_contiguous().reshape(new_shape)
        }
    }

    /// Transpose two dimensions. Zero-copy (metadata only).
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.transpose(dim1, dim2),
            dtype: self.dtype,
        }
    }

    /// Narrow/slice along a dimension. Zero-copy (metadata only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.narrow(dim, start, len),
            dtype: self.dtype,
        }
    }

    /// Permute dimensions according to axes. Zero-copy (metadata only).
    pub fn permute(&self, axes: &[usize]) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.permute(axes),
            dtype: self.dtype,
        }
    }
}

impl TensorMetadata for FlexTensor {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.layout.shape().clone()
    }

    fn rank(&self) -> usize {
        self.layout.num_dims()
    }
}

/// Get the size in bytes for a dtype element.
///
/// Matches `burn_std::DType::size()` semantics: Bool(Native) and Bool(U8) are
/// 1 byte, Bool(U32) is 4 bytes. This makes buffer-size validation correct
/// regardless of which BoolStore variant the dtype carries.
pub(crate) fn dtype_size(dtype: DType) -> usize {
    // Delegate to burn-std's canonical size to stay in sync.
    dtype.size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_from_data_roundtrip() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor = FlexTensor::from_data(data.clone());
        let result = tensor.into_data();
        assert_eq!(data.shape, result.shape);
        assert_eq!(data.dtype, result.dtype);
    }

    #[test]
    fn test_reshape() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        let reshaped = tensor.reshape(Shape::from(vec![3, 2]));
        assert_eq!(reshaped.shape().to_vec(), vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        let transposed = tensor.transpose(0, 1);
        assert_eq!(transposed.shape().to_vec(), vec![3, 2]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_clone_is_cheap() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = FlexTensor::from_data(data);

        // Before clone, tensor is unique
        assert!(tensor.is_unique());

        // Clone shares data
        let cloned = tensor.clone();
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Both point to same data
        assert!(core::ptr::eq(
            tensor.bytes().as_ptr(),
            cloned.bytes().as_ptr()
        ));
    }

    #[test]
    fn test_cow_on_mutation() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = FlexTensor::from_data(data);
        let mut cloned = tensor.clone();

        // Both share data
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Mutate cloned - triggers COW
        let storage: &mut [f32] = cloned.storage_mut();
        storage[0] = 99.0;

        // Now cloned has its own copy, tensor is unique again
        assert!(tensor.is_unique());
        assert!(cloned.is_unique());

        // Data is different
        assert_ne!(tensor.bytes().as_ptr(), cloned.bytes().as_ptr());
        assert_eq!(tensor.storage::<f32>()[0], 1.0);
        assert_eq!(cloned.storage::<f32>()[0], 99.0);
    }

    #[test]
    fn test_into_data_narrowed_at_offset_zero() {
        // [1, 2, 3, 4, 5, 6] shape [2, 3]
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        // narrow to first row: shape [1, 3], offset 0, contiguous
        let narrowed = tensor.narrow(0, 0, 1);
        assert!(narrowed.is_contiguous());
        assert_eq!(narrowed.layout().start_offset(), 0);

        let result = narrowed.into_data();
        assert_eq!(result.shape.to_vec(), vec![1, 3]);
        // Must have exactly 3 f32s = 12 bytes, not 24
        assert_eq!(result.bytes.len(), 3 * core::mem::size_of::<f32>());
        let values: Vec<f32> = result.to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
