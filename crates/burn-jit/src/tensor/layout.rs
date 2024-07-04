#[derive(PartialEq, Eq, Debug)]
pub(crate) enum MemoryLayout {
    /// Memory is wholy contiguous, with row major layout
    Contiguous,
    /// Permutations happened, but may not impact some kernels
    MildlyPermuted {
        /// Last two dims are inverted
        transposed: bool,
        /// Some permutations exist in batch dimensions
        batch_swap: bool,
    },
    /// Permutations happened between batch dimensions and last two dims
    HighlyPermuted,
}

pub(crate) fn memory_layout<const D: usize>(strides: &[usize; D]) -> MemoryLayout {
    if D <= 1 {
        return MemoryLayout::Contiguous;
    }

    let mut transposed = false;
    let mut batch_swap = false;
    let row_stride = strides[D - 2];
    let col_stride = strides[D - 1];
    if row_stride < col_stride {
        transposed = true;
    }
    let mut previous_stride = row_stride;

    for d in 0..D - 2 {
        let current_stride = strides[D - 3 - d];
        if current_stride < row_stride || current_stride < col_stride {
            return MemoryLayout::HighlyPermuted;
        }
        if current_stride < previous_stride {
            batch_swap = true;
        }

        previous_stride = current_stride;
    }

    if transposed || batch_swap {
        MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap,
        }
    } else {
        MemoryLayout::Contiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_contiguous() {
        let strides = &[8, 4, 2, 1];
        assert_eq!(memory_layout(strides), MemoryLayout::Contiguous);
    }

    #[test]
    fn vector_is_contiguous() {
        let strides = &[1];
        assert_eq!(memory_layout(strides), MemoryLayout::Contiguous)
    }

    #[test]
    fn layout_is_transposed_only() {
        let strides = &[8, 4, 1, 2];
        if let MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = memory_layout(strides)
        {
            assert!(transposed && !batch_swap);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn layout_has_swapped_batches_only() {
        let strides = &[4, 8, 2, 1];
        if let MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = memory_layout(strides)
        {
            assert!(!transposed && batch_swap);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn layout_has_swapped_batches_and_is_transposed() {
        let strides = &[4, 8, 1, 2];
        if let MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } = memory_layout(strides)
        {
            assert!(transposed && batch_swap);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn layout_has_batch_swapped_with_row() {
        let strides = &[8, 2, 4, 1];
        assert_eq!(memory_layout(strides), MemoryLayout::HighlyPermuted);
    }

    #[test]
    fn layout_has_batch_swapped_with_col() {
        let strides = &[1, 4, 2, 8];
        assert_eq!(memory_layout(strides), MemoryLayout::HighlyPermuted);
    }
}
