use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Shape, Slice};

/// Layout describes how to interpret a linear buffer as an N-dimensional tensor.
///
/// Stores shape, strides (in elements, can be negative for flipped dimensions),
/// and a start offset for views/slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    /// Strides in elements. Negative strides enable zero-copy flip.
    strides: Vec<isize>,
    start_offset: usize,
}

/// Compute row-major contiguous strides for a shape (as `usize`).
pub(crate) fn contiguous_strides_usize(shape: &Shape) -> Vec<usize> {
    let ndims = shape.num_dims();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the flat offset for the `slice_idx`-th 1D fiber along `dim`.
///
/// Enumerates all index combinations for dimensions other than `dim`,
/// mapping the flat `slice_idx` (0..product of non-dim sizes) to the
/// corresponding starting offset in a contiguous buffer.
pub(crate) fn slice_base_offset(
    slice_idx: usize,
    shape: &Shape,
    strides: &[usize],
    dim: usize,
) -> usize {
    let ndims = shape.num_dims();
    let mut offset = 0;
    let mut remaining = slice_idx;
    for d in (0..ndims).rev() {
        if d == dim {
            continue;
        }
        let s = shape[d];
        offset += (remaining % s) * strides[d];
        remaining /= s;
    }
    offset
}

impl Layout {
    /// Create a new contiguous layout (row-major/C-order).
    pub fn contiguous(shape: Shape) -> Self {
        let strides: Vec<isize> = contiguous_strides_usize(&shape)
            .into_iter()
            .map(|s| s as isize)
            .collect();

        Self {
            shape,
            strides,
            start_offset: 0,
        }
    }

    /// Create a layout with explicit strides.
    pub fn new(shape: Shape, strides: Vec<isize>, start_offset: usize) -> Self {
        debug_assert_eq!(shape.num_dims(), strides.len());
        Self {
            shape,
            strides,
            start_offset,
        }
    }

    /// The shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The strides in elements (can be negative for flipped dimensions).
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// The start offset for views/slices.
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Number of dimensions.
    pub fn num_dims(&self) -> usize {
        self.shape.num_dims()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Check if this layout is contiguous (row-major, positive strides).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.num_dims() == 0 {
            return true;
        }

        let mut expected_stride = 1isize;
        for i in (0..self.shape.num_dims()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// If contiguous, return (start, end) offsets for direct slice access.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            Some((self.start_offset, self.start_offset + self.num_elements()))
        } else {
            None
        }
    }

    /// Transpose: swap two dimensions (zero-copy, metadata only).
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut dims = self.shape.to_vec();
        let mut strides = self.strides.clone();
        dims.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Self {
            shape: Shape::from(dims),
            strides,
            start_offset: self.start_offset,
        }
    }

    /// Permute: reorder dimensions according to axes (zero-copy, metadata only).
    ///
    /// `axes` must be a permutation of 0..ndim.
    pub fn permute(&self, axes: &[usize]) -> Self {
        debug_assert_eq!(
            axes.len(),
            self.num_dims(),
            "permute: axes length must match number of dimensions"
        );

        let new_dims: Vec<usize> = axes.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<isize> = axes.iter().map(|&i| self.strides[i]).collect();

        Self {
            shape: Shape::from(new_dims),
            strides: new_strides,
            start_offset: self.start_offset,
        }
    }

    /// Flip: reverse elements along specified axes (zero-copy, metadata only).
    ///
    /// For each flipped axis, negates the stride and adjusts start_offset
    /// to point to the last element along that dimension.
    pub fn flip(&self, axes: &[usize]) -> Self {
        let mut new_strides = self.strides.clone();
        let mut offset_adjustment: isize = 0;

        for &axis in axes {
            debug_assert!(
                axis < self.num_dims(),
                "flip: axis {} out of bounds for {} dimensions",
                axis,
                self.num_dims()
            );

            let dim_size = self.shape[axis];
            if dim_size > 1 {
                // Move start to last element along this axis
                offset_adjustment += (dim_size as isize - 1) * self.strides[axis];
                // Negate stride to iterate backwards
                new_strides[axis] = -new_strides[axis];
            }
        }

        let new_start_isize = self.start_offset as isize + offset_adjustment;
        debug_assert!(new_start_isize >= 0, "flip: negative offset");
        let new_start = new_start_isize as usize;

        Self {
            shape: self.shape.clone(),
            strides: new_strides,
            start_offset: new_start,
        }
    }

    /// Narrow/slice along a dimension (zero-copy, metadata only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        debug_assert!(
            start + len <= self.shape[dim],
            "narrow: start ({}) + len ({}) exceeds dimension size ({})",
            start,
            len,
            self.shape[dim]
        );
        let mut dims = self.shape.to_vec();
        dims[dim] = len;

        let new_offset_isize = self.start_offset as isize + self.strides[dim] * start as isize;
        debug_assert!(new_offset_isize >= 0, "narrow: negative offset");
        let new_offset = new_offset_isize as usize;

        Self {
            shape: Shape::from(dims),
            strides: self.strides.clone(),
            start_offset: new_offset,
        }
    }

    /// Apply slices to create a new layout.
    ///
    /// Returns `(new_layout, needs_copy)`:
    /// - `needs_copy = false`: Can use zero-copy view with new layout
    /// - `needs_copy = true`: Has negative steps requiring data reordering
    pub fn slice(&self, slices: &[Slice]) -> (Self, bool) {
        let ndims = self.num_dims();
        let mut new_dims = self.shape.to_vec();
        let mut new_strides = self.strides.clone();
        let mut new_offset = self.start_offset as isize;
        let mut needs_copy = false;

        for (dim, slice) in slices.iter().enumerate() {
            if dim >= ndims {
                break;
            }

            let dim_size = self.shape[dim] as isize;
            let stride = self.strides[dim];

            // Normalize start index (handle negative)
            let start = if slice.start < 0 {
                (dim_size + slice.start).max(0) as usize
            } else {
                (slice.start as usize).min(dim_size as usize)
            };

            // Normalize end index (handle negative and None)
            // Note: Range [start, end) determines WHICH elements to select,
            // step determines iteration ORDER
            let end = match slice.end {
                Some(e) if e < 0 => (dim_size + e).max(0) as usize,
                Some(e) => (e as usize).min(dim_size as usize),
                None => dim_size as usize, // Always full range when end is None
            };

            let step = slice.step;
            let abs_step = step.unsigned_abs();

            if step > 0 {
                // Positive step: forward iteration
                let len = if end > start {
                    (end - start).div_ceil(abs_step)
                } else {
                    0
                };
                new_dims[dim] = len;
                new_strides[dim] = stride * step;
                new_offset += stride * start as isize;
            } else {
                // Negative step: select range then iterate in reverse
                // Requires copy to reorder elements
                needs_copy = true;
                let len = if end > start {
                    (end - start).div_ceil(abs_step)
                } else {
                    0
                };
                new_dims[dim] = len;
                new_strides[dim] = stride; // Will be handled during copy
            }
        }

        debug_assert!(new_offset >= 0, "slice: negative offset");

        (
            Self {
                shape: Shape::from(new_dims),
                strides: new_strides,
                start_offset: new_offset as usize,
            },
            needs_copy,
        )
    }

    /// Reshape to a new shape. Only works if contiguous with zero offset.
    ///
    /// Returns None if not contiguous or has non-zero offset (would require data copy).
    pub fn reshape(&self, new_shape: Shape) -> Option<Self> {
        if !self.is_contiguous() || self.start_offset != 0 {
            return None;
        }
        debug_assert_eq!(
            self.num_elements(),
            new_shape.num_elements(),
            "reshape must preserve total elements"
        );
        Some(Self::contiguous(new_shape))
    }

    /// Compute linear index from multi-dimensional indices.
    pub fn index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.num_dims());
        let mut offset = self.start_offset as isize;
        for (i, &idx) in indices.iter().enumerate() {
            offset += idx as isize * self.strides[i];
        }
        debug_assert!(offset >= 0, "index: negative offset");
        offset as usize
    }

    /// Get stride of the innermost (last) dimension.
    /// Returns 1 for contiguous tensors, larger values for transposed.
    /// Returns absolute value (ignores flip).
    pub fn inner_stride(&self) -> usize {
        self.strides.last().map(|s| s.unsigned_abs()).unwrap_or(1)
    }

    /// Check if innermost dimension is contiguous (|stride| == 1).
    /// This enables efficient vectorized inner loops.
    pub fn has_contiguous_inner(&self) -> bool {
        self.inner_stride() == 1
    }

    /// For 2D layouts, get (outer_size, inner_size, outer_stride, inner_stride).
    /// Returns None if not 2D.
    pub fn as_2d_strides(&self) -> Option<(usize, usize, isize, isize)> {
        if self.num_dims() != 2 {
            return None;
        }
        Some((
            self.shape[0],
            self.shape[1],
            self.strides[0],
            self.strides[1],
        ))
    }

    /// Check if all strides are non-negative.
    pub fn has_positive_strides(&self) -> bool {
        self.strides.iter().all(|&s| s >= 0)
    }

    /// Compute strided blocks for efficient iteration.
    ///
    /// Returns (block_len, num_blocks, block_stride) where:
    /// - block_len: number of contiguous elements in each block
    /// - num_blocks: total number of blocks
    /// - block_stride: stride between consecutive blocks (0 if single block)
    ///
    /// For contiguous tensors: single block covering all elements.
    /// For transposed/strided: multiple blocks of contiguous data.
    pub fn strided_blocks(&self) -> StridedBlocks<'_> {
        let n = self.num_elements();
        if n == 0 {
            return StridedBlocks::Single { start: 0, len: 0 };
        }

        // Fast path: fully contiguous
        if self.is_contiguous() {
            return StridedBlocks::Single {
                start: self.start_offset,
                len: n,
            };
        }

        // Find contiguous inner dimensions (only positive strides)
        // Start from innermost and work outward while strides match contiguous pattern
        let ndims = self.num_dims();
        let mut block_len = 1usize;
        let mut expected_stride = 1isize;

        for i in (0..ndims).rev() {
            if self.strides[i] == expected_stride {
                block_len *= self.shape[i];
                expected_stride *= self.shape[i] as isize;
            } else {
                break;
            }
        }

        if block_len == n {
            // All dimensions contiguous (just offset)
            return StridedBlocks::Single {
                start: self.start_offset,
                len: n,
            };
        }

        let num_blocks = n / block_len;
        StridedBlocks::Multiple {
            layout: self,
            block_len,
            num_blocks,
        }
    }
}

/// Result of strided block analysis.
#[derive(Debug, Clone)]
pub enum StridedBlocks<'a> {
    /// Single contiguous block - direct slice access.
    Single { start: usize, len: usize },
    /// Multiple blocks requiring iteration.
    Multiple {
        layout: &'a Layout,
        block_len: usize,
        num_blocks: usize,
    },
}

impl<'a> StridedBlocks<'a> {
    /// Get the block length (elements per block).
    pub fn block_len(&self) -> usize {
        match self {
            Self::Single { len, .. } => *len,
            Self::Multiple { block_len, .. } => *block_len,
        }
    }

    /// Iterator over block start indices.
    pub fn block_starts(&self) -> BlockStartIter<'_> {
        match self {
            Self::Single { start, .. } => BlockStartIter::Single {
                start: *start,
                done: false,
            },
            Self::Multiple {
                layout,
                block_len,
                num_blocks,
            } => {
                // Calculate dimensions for outer iteration (non-contiguous part)
                let ndims = layout.num_dims();
                let mut outer_dims = 0;
                let mut expected_stride = 1isize;

                for i in (0..ndims).rev() {
                    if layout.strides[i] == expected_stride {
                        expected_stride *= layout.shape[i] as isize;
                    } else {
                        outer_dims = i + 1;
                        break;
                    }
                }

                BlockStartIter::Multiple {
                    layout,
                    multi_index: vec![0; outer_dims],
                    remaining: *num_blocks,
                    block_len: *block_len,
                }
            }
        }
    }
}

/// Iterator over block start indices.
pub enum BlockStartIter<'a> {
    Single {
        start: usize,
        done: bool,
    },
    Multiple {
        layout: &'a Layout,
        multi_index: Vec<usize>,
        remaining: usize,
        block_len: usize,
    },
}

impl Iterator for BlockStartIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Single { start, done } => {
                if *done {
                    None
                } else {
                    *done = true;
                    Some(*start)
                }
            }
            Self::Multiple {
                layout,
                multi_index,
                remaining,
                block_len: _,
            } => {
                if *remaining == 0 {
                    return None;
                }

                // Compute current block start
                let outer_dims = multi_index.len();
                let mut offset = layout.start_offset as isize;
                for (i, &idx) in multi_index.iter().enumerate() {
                    offset += idx as isize * layout.strides[i];
                }

                *remaining -= 1;

                // Advance multi-index for next iteration
                let shape = &layout.shape;
                for d in (0..outer_dims).rev() {
                    multi_index[d] += 1;
                    if multi_index[d] < shape[d] {
                        break;
                    }
                    multi_index[d] = 0;
                }

                Some(offset as usize)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Single { done, .. } => {
                if *done {
                    0
                } else {
                    1
                }
            }
            Self::Multiple { remaining, .. } => *remaining,
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for BlockStartIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_layout() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3, 4]));
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_transpose() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let transposed = layout.transpose(0, 1);
        assert_eq!(transposed.shape().to_vec(), vec![3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_narrow() {
        let layout = Layout::contiguous(Shape::from(vec![4, 4]));
        let narrowed = layout.narrow(0, 1, 2);
        assert_eq!(narrowed.shape().to_vec(), vec![2, 4]);
        assert_eq!(narrowed.start_offset(), 4);
    }

    #[test]
    fn test_contiguous_offsets() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        assert_eq!(layout.contiguous_offsets(), Some((0, 6)));
    }

    #[test]
    fn test_index() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        assert_eq!(layout.index(&[0, 0]), 0);
        assert_eq!(layout.index(&[0, 2]), 2);
        assert_eq!(layout.index(&[1, 0]), 3);
        assert_eq!(layout.index(&[1, 2]), 5);
    }

    #[test]
    fn test_flip_1d() {
        // Original: [0, 1, 2, 3] with strides [1]
        // Flipped: strides [-1], start_offset = 3
        let layout = Layout::contiguous(Shape::from(vec![4]));
        let flipped = layout.flip(&[0]);

        assert_eq!(flipped.shape().to_vec(), vec![4]);
        assert_eq!(flipped.strides(), &[-1]);
        assert_eq!(flipped.start_offset(), 3);

        // Verify indices: logical [0] -> physical [3], logical [1] -> physical [2], etc.
        assert_eq!(flipped.index(&[0]), 3);
        assert_eq!(flipped.index(&[1]), 2);
        assert_eq!(flipped.index(&[2]), 1);
        assert_eq!(flipped.index(&[3]), 0);
    }

    #[test]
    fn test_flip_2d_axis0() {
        // [[0, 1, 2], [3, 4, 5]] with strides [3, 1]
        // Flip axis 0: strides [-3, 1], start_offset = 3
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let flipped = layout.flip(&[0]);

        assert_eq!(flipped.strides(), &[-3, 1]);
        assert_eq!(flipped.start_offset(), 3);

        // Row 0 of flipped = Row 1 of original
        assert_eq!(flipped.index(&[0, 0]), 3);
        assert_eq!(flipped.index(&[0, 1]), 4);
        assert_eq!(flipped.index(&[0, 2]), 5);
        // Row 1 of flipped = Row 0 of original
        assert_eq!(flipped.index(&[1, 0]), 0);
        assert_eq!(flipped.index(&[1, 1]), 1);
        assert_eq!(flipped.index(&[1, 2]), 2);
    }

    #[test]
    fn test_flip_2d_axis1() {
        // [[0, 1, 2], [3, 4, 5]] with strides [3, 1]
        // Flip axis 1: strides [3, -1], start_offset = 2
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let flipped = layout.flip(&[1]);

        assert_eq!(flipped.strides(), &[3, -1]);
        assert_eq!(flipped.start_offset(), 2);

        // Col 0 of flipped = Col 2 of original
        assert_eq!(flipped.index(&[0, 0]), 2);
        assert_eq!(flipped.index(&[0, 1]), 1);
        assert_eq!(flipped.index(&[0, 2]), 0);
        assert_eq!(flipped.index(&[1, 0]), 5);
        assert_eq!(flipped.index(&[1, 1]), 4);
        assert_eq!(flipped.index(&[1, 2]), 3);
    }

    #[test]
    fn test_flip_both_axes() {
        // [[0, 1, 2], [3, 4, 5]] -> [[5, 4, 3], [2, 1, 0]]
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let flipped = layout.flip(&[0, 1]);

        assert_eq!(flipped.strides(), &[-3, -1]);
        assert_eq!(flipped.start_offset(), 5); // 3 + 2 = 5

        assert_eq!(flipped.index(&[0, 0]), 5);
        assert_eq!(flipped.index(&[0, 1]), 4);
        assert_eq!(flipped.index(&[0, 2]), 3);
        assert_eq!(flipped.index(&[1, 0]), 2);
        assert_eq!(flipped.index(&[1, 1]), 1);
        assert_eq!(flipped.index(&[1, 2]), 0);
    }
}
