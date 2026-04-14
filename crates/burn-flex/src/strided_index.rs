use crate::layout::Layout;
use alloc::vec;
use alloc::vec::Vec;

/// Iterator that yields linear indices for strided tensor access.
///
/// Handles non-contiguous tensors (including those with negative strides from flip)
/// by computing the correct storage index for each logical element position.
pub struct StridedIter<'a> {
    /// Current storage index (signed to handle negative strides)
    storage_index: isize,
    multi_index: Vec<usize>,
    layout: &'a Layout,
    remaining: usize,
}

impl<'a> StridedIter<'a> {
    /// Create a new strided iterator for the given layout.
    pub fn new(layout: &'a Layout) -> Self {
        let ndims = layout.num_dims();
        Self {
            storage_index: layout.start_offset() as isize,
            multi_index: vec![0; ndims],
            layout,
            remaining: layout.num_elements(),
        }
    }
}

impl Iterator for StridedIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }

        debug_assert!(
            self.storage_index >= 0,
            "StridedIter: negative storage index"
        );
        let idx = self.storage_index as usize;
        self.remaining -= 1;

        // Advance multi-index (last dimension first, like odometer)
        let shape = self.layout.shape();
        let strides = self.layout.strides();

        for d in (0..shape.num_dims()).rev() {
            self.multi_index[d] += 1;
            if self.multi_index[d] < shape[d] {
                self.storage_index += strides[d];
                break;
            }
            // Wrap around this dimension
            self.multi_index[d] = 0;
            self.storage_index -= (shape[d] as isize - 1) * strides[d];
        }

        Some(idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_std::Shape;

    #[test]
    fn test_contiguous_iteration() {
        let layout = Layout::contiguous(Shape::from(vec![2, 3]));
        let indices: Vec<_> = StridedIter::new(&layout).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_transposed_iteration() {
        // Original: [[0, 1, 2], [3, 4, 5]] (shape [2, 3], strides [3, 1])
        // Transposed: [[0, 3], [1, 4], [2, 5]] (shape [3, 2], strides [1, 3])
        let layout = Layout::contiguous(Shape::from(vec![2, 3])).transpose(0, 1);
        let indices: Vec<_> = StridedIter::new(&layout).collect();
        assert_eq!(indices, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn test_narrowed_iteration() {
        // Original: [[0, 1, 2, 3], [4, 5, 6, 7]] (shape [2, 4])
        // Narrowed dim=1, start=1, len=2: [[1, 2], [5, 6]]
        let layout = Layout::contiguous(Shape::from(vec![2, 4])).narrow(1, 1, 2);
        let indices: Vec<_> = StridedIter::new(&layout).collect();
        assert_eq!(indices, vec![1, 2, 5, 6]);
    }
}
