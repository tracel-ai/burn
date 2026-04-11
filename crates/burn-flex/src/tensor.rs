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
    /// Note: Bool tensors are stored as u8, so both Bool(Native) and Bool(U8)
    /// dtypes accept u8 access.
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
    /// Note: Bool tensors are stored as u8, so both Bool(Native) and Bool(U8)
    /// dtypes accept u8 access.
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
    /// Note: Bool tensors are stored as u8, so both Bool(Native) and Bool(U8)
    /// dtypes accept u8 access.
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

        // Squeeze size-1 dims and merge adjacent stride-contiguous
        // runs so e.g. a permuted `[N, H, W, C]` ConvNeXt layer-norm
        // input becomes a plain 2D `[H*W, C]` transpose that the
        // tiled copy below handles at near-memcpy speed. Without the
        // collapse, the 4D ND fallback scalar-walks the tensor.
        let collapsed = collapse_for_copy(self.layout.shape(), self.layout.strides());
        let (shape, strides) = collapsed.as_slices();
        let offset = self.layout.start_offset() as isize;
        let all_positive = strides.iter().all(|&s| s >= 0);

        if shape.len() <= 1 && all_positive {
            // 0-D scalar or 1-D run with a uniform stride. Empty
            // collapsed shape means rank 0 (numel 1); otherwise
            // numel is the single dim's size (which may be 0 for
            // zero-sized 1D tensors, so don't clamp via `.max(1)`).
            let collapsed_numel = if shape.is_empty() { 1 } else { shape[0] };
            debug_assert_eq!(n, collapsed_numel);
            // SAFETY: capacity is n; we fill every position below.
            unsafe { dst.set_len(n) };
            if shape.is_empty() {
                if n > 0 {
                    dst[0] = src[offset as usize];
                }
            } else {
                let len = shape[0];
                let stride = strides[0];
                if stride == 1 {
                    dst[..len].copy_from_slice(&src[offset as usize..offset as usize + len]);
                } else {
                    for (i, slot) in dst.iter_mut().take(len).enumerate() {
                        let idx = (offset + i as isize * stride) as usize;
                        *slot = src[idx];
                    }
                }
            }
        } else if shape.len() == 2 && all_positive {
            // 2D positive-stride (transpose-like): tile both dims so
            // reads stay in cache. The loop-nesting chooser inside
            // `copy_2d_tiled` picks whichever ordering puts the
            // smaller source stride on the innermost loop.
            debug_assert_eq!(shape[0] * shape[1], n, "2D strides must cover all elements");
            // SAFETY: capacity is n; `copy_2d_tiled` writes every
            // `(row, col)` position exactly once.
            unsafe { dst.set_len(n) };
            copy_2d_tiled(
                &mut dst, src, offset, shape[0], shape[1], strides[0], strides[1],
            );
        } else {
            // General fallback: covers negative strides (flipped
            // tensors) and ND layouts that can't collapse to ≤2D.
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

/// Max rank we're willing to handle without falling back to the
/// strided iterator. Burn tensors are capped at 8 dims in practice.
const COLLAPSE_MAX_RANK: usize = 8;

/// Collapsed layout result of [`collapse_for_copy`], stored in stack
/// arrays so `to_contiguous()` doesn't have to hit the allocator on
/// its hot path.
#[derive(Debug, Clone, Copy)]
struct CollapsedLayout {
    ndim: usize,
    shape: [usize; COLLAPSE_MAX_RANK],
    strides: [isize; COLLAPSE_MAX_RANK],
}

impl CollapsedLayout {
    #[inline]
    fn as_slices(&self) -> (&[usize], &[isize]) {
        (&self.shape[..self.ndim], &self.strides[..self.ndim])
    }
}

/// Collapse a shape/stride pair into the minimum-rank equivalent
/// layout for a contiguous copy:
///
/// 1. Squeeze size-1 dims (their stride never gets stepped past 0).
/// 2. Merge adjacent dims `(i, i+1)` when
///    `stride[i] == stride[i+1] * shape[i+1]`, which means the two
///    dims form a single logical run through memory.
///
/// Canonical example: a 4D ConvNeXt input `[1, 244, 224, 48]` with
/// strides `[2_623_488, 224, 1, 54656]` (from
/// `[N, C, H, W].permute([0, 2, 3, 1])`) collapses to 2D
/// `[54656, 48]` with strides `[1, 54656]`.
///
/// If the input rank exceeds [`COLLAPSE_MAX_RANK`] the result is
/// left at rank > 2 so the caller falls through to its generic
/// strided path. If the input is rank > `COLLAPSE_MAX_RANK`, we
/// return the original (un-collapsed) layout truncated, which the
/// caller will reject via its `shape.len() == 2` gate.
///
/// PRECONDITION: the caller must gate on all-positive strides before
/// using the collapsed layout. The merge rule assumes positive
/// strides and will produce iteration-order-incorrect results for
/// flipped tensors.
fn collapse_for_copy(shape: &[usize], strides: &[isize]) -> CollapsedLayout {
    let mut out = CollapsedLayout {
        ndim: 0,
        shape: [0; COLLAPSE_MAX_RANK],
        strides: [0; COLLAPSE_MAX_RANK],
    };

    // Bail out to the caller's fallback if the rank is too large to
    // fit our stack buffer. In practice this never triggers (burn
    // tensors are ≤8 dims), but leaving the `ndim` high signals the
    // caller to take the generic strided path.
    if shape.len() > COLLAPSE_MAX_RANK {
        out.ndim = shape.len().min(COLLAPSE_MAX_RANK);
        return out;
    }

    // Single forward sweep: squeeze size-1 dims and merge whenever
    // the current dim's `stride * size` equals the previous output
    // dim's stride (i.e. the two form a contiguous run).
    //
    // Use `checked_mul` so a pathological layout whose stride math
    // would overflow `isize` simply fails to merge rather than
    // wrapping into an incorrect merge decision. Real tensors can't
    // hit this (total numel is bounded by `isize::MAX`), but
    // hand-built layouts passed through the test paths could.
    for (&s, &st) in shape.iter().zip(strides.iter()) {
        if s == 1 {
            continue;
        }
        let merge = out.ndim > 0
            && (s as isize)
                .checked_mul(st)
                .is_some_and(|run| out.strides[out.ndim - 1] == run);
        if merge {
            out.shape[out.ndim - 1] *= s;
            out.strides[out.ndim - 1] = st;
        } else {
            out.shape[out.ndim] = s;
            out.strides[out.ndim] = st;
            out.ndim += 1;
        }
    }

    out
}

/// Tiled 2D copy from a strided source into a contiguous destination.
/// The loop nesting is chosen so the innermost read walks whichever
/// source stride is smaller, which keeps the hot loop in cache even
/// for transpose-like layouts.
#[inline]
fn copy_2d_tiled<E: Copy>(
    dst: &mut [E],
    src: &[E],
    offset: isize,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
) {
    const TILE: usize = 16;

    if row_stride <= col_stride {
        // row-inside-col: the inner loop walks `row_stride` (smaller).
        for col_tile in (0..cols).step_by(TILE) {
            let col_end = (col_tile + TILE).min(cols);
            for row_tile in (0..rows).step_by(TILE) {
                let row_end = (row_tile + TILE).min(rows);
                for col in col_tile..col_end {
                    let col_base = offset + col as isize * col_stride;
                    for row in row_tile..row_end {
                        let idx = (col_base + row as isize * row_stride) as usize;
                        // SAFETY: caller set `dst.len() == rows * cols`
                        // and each `(row, col)` is visited once.
                        unsafe {
                            *dst.get_unchecked_mut(row * cols + col) = src[idx];
                        }
                    }
                }
            }
        }
    } else {
        // col-inside-row: the inner loop walks `col_stride` (smaller).
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
                        // SAFETY: same as above.
                        unsafe {
                            *dst.get_unchecked_mut(dst_base + c) = src[idx];
                        }
                    }
                }
            }
        }
    }
}

/// Get the size in bytes for a dtype element.
///
/// Matches `burn_std::DType::size()` semantics: Bool(Native) and Bool(U8) are
/// 1 byte, Bool(U32) is 4 bytes. This makes buffer-size validation correct
/// regardless of which BoolStore variant the dtype carries.
///
/// # Panics
///
/// Panics if the dtype has a zero-byte element size. `burn_std::DType::size()`
/// returns 0 for sub-byte quantized dtypes (Q4F, Q4S, Q2F, Q2S, and most
/// `QuantStore::PackedNative` variants). burn-flex does not yet support these
/// packed quantization formats; passing them here would silently produce
/// empty allocations in `FlexTensor::empty`, truncated buffers in `into_data`,
/// and zero-byte memcpys in `repeat_dim`. The panic turns all three into a
/// loud, actionable failure at the dispatch boundary.
pub(crate) fn dtype_size(dtype: DType) -> usize {
    // Delegate to burn-std's canonical size to stay in sync.
    let size = dtype.size();
    assert!(
        size > 0,
        "burn-flex: dtype {:?} has zero-byte element size (sub-byte packed \
         quantization is not yet supported)",
        dtype
    );
    size
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
    fn test_collapse_for_copy_squeezes_size1_and_merges_contig() {
        // Permuted ConvNeXt input: [1, 48, 244, 224].permute([0,2,3,1]).
        let shape = vec![1, 244, 224, 48];
        let strides = vec![2_623_488_isize, 224, 1, 54656];
        let collapsed = collapse_for_copy(&shape, &strides);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[54656, 48]);
        assert_eq!(st, &[1, 54656]);
    }

    #[test]
    fn test_collapse_for_copy_already_contiguous_3d() {
        let collapsed = collapse_for_copy(&[2, 3, 4], &[12, 4, 1]);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[24]);
        assert_eq!(st, &[1]);
    }

    #[test]
    fn test_collapse_for_copy_transpose_2d() {
        let collapsed = collapse_for_copy(&[5, 3], &[1, 5]);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[5, 3]);
        assert_eq!(st, &[1, 5]);
    }

    #[test]
    fn test_collapse_for_copy_all_size1() {
        let collapsed = collapse_for_copy(&[1, 1, 1], &[0, 0, 0]);
        let (s, st) = collapsed.as_slices();
        assert!(s.is_empty());
        assert!(st.is_empty());
    }

    /// Regression: an empty 1D view produced by `narrow` at a
    /// non-zero offset forces `copy_contiguous` to run (it can't
    /// early-return via the contiguous-at-offset-0 shortcut). The
    /// old `debug_assert_eq!(n, shape.product().max(1))` tripped
    /// for this shape because `.max(1)` produced 1 while the true
    /// numel is 0.
    #[test]
    fn test_to_contiguous_zero_sized_narrowed() {
        let t = FlexTensor::from_data(TensorData::new(
            (0..6).map(|i| i as f32).collect::<Vec<_>>(),
            vec![6],
        ));
        // narrow(dim, start=3, len=0): shape [0], start_offset 3.
        let empty_view = t.narrow(0, 3, 0);
        assert_eq!(empty_view.shape().to_vec(), vec![0]);
        assert_ne!(empty_view.layout().start_offset(), 0);

        let contig = empty_view.to_contiguous();
        assert_eq!(contig.shape().to_vec(), vec![0]);
        assert_eq!(contig.layout().start_offset(), 0);
        assert_eq!(contig.into_data().bytes.len(), 0);
    }

    /// 4D permuted layout round-trips through the collapse + tiled
    /// copy path. Mirrors the ConvNeXt channels-last permute.
    #[test]
    fn test_to_contiguous_4d_permuted_matches_naive() {
        let dims = [1, 48, 4, 5];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = FlexTensor::from_data(TensorData::new(data.clone(), dims.to_vec()));
        let permuted = t.permute(&[0, 2, 3, 1]);
        assert!(!permuted.is_contiguous());

        let contig = permuted.to_contiguous();
        assert!(contig.is_contiguous());
        assert_eq!(contig.shape().to_vec(), vec![1, 4, 5, 48]);

        // Expected via manual strided walk of the source.
        let mut expected = Vec::with_capacity(n);
        for h in 0..4 {
            for w in 0..5 {
                for c in 0..48 {
                    let idx = c * 20 + h * 5 + w;
                    expected.push(data[idx]);
                }
            }
        }

        let result_data = contig.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        assert_eq!(values, expected.as_slice());
    }

    /// Exercise the `row_stride > col_stride` branch of the 2D tiled
    /// copy (the ConvNeXt case hits the other branch).
    #[test]
    fn test_to_contiguous_2d_row_stride_gt_col_stride() {
        // `slice(s![..;2, ..])` on a [6, 3] contiguous tensor gives a
        // [3, 3] view with strides [6, 1] that doesn't collapse, so
        // the 2D branch runs with row_stride > col_stride.
        let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let t = FlexTensor::from_data(TensorData::new(data, vec![6, 3]));
        let stepped = crate::ops::slice::slice(
            t,
            &[
                burn_std::Slice::new(0, Some(6), 2),
                burn_std::Slice::new(0, None, 1),
            ],
        );
        // Verify the layout matches what the branch requires.
        assert_eq!(stepped.layout().shape().to_vec(), vec![3, 3]);
        assert_eq!(stepped.layout().strides(), &[6, 1]);
        assert!(!stepped.layout().is_contiguous());

        let contig = stepped.to_contiguous();
        assert!(contig.is_contiguous());
        assert_eq!(contig.shape().to_vec(), vec![3, 3]);

        let result_data = contig.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        // Expected: rows 0, 2, 4 of the original 6x3 tensor.
        let expected = vec![
            0.0f32, 1.0, 2.0, // row 0
            6.0, 7.0, 8.0, // row 2
            12.0, 13.0, 14.0, // row 4
        ];
        assert_eq!(values, expected.as_slice());
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
