use alloc::vec::Vec;

use crate::indexing::AsIndex;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Calculates the output shape for a slice operation with steps
///
/// # Arguments
/// * `slices` - The slice specifications for each dimension
/// * `original_shape` - The original tensor shape
///
/// # Returns
/// The output shape after applying the slice operation
pub fn calculate_slice_output_shape(slices: &[Slice], original_shape: &[usize]) -> Vec<usize> {
    let mut shape: Vec<usize> = slices
        .iter()
        .zip(original_shape.iter())
        .map(|(slice, &dim_size)| slice.output_size(dim_size))
        .collect();

    // Add remaining dimensions from original shape
    shape.extend_from_slice(&original_shape[shape.len()..]);

    shape
}

/// Creates a slice specification for tensor indexing operations.
///
/// This macro simplifies the creation of tensor slices by allowing various range types
/// to be used together in a concise way. It supports all standard Rust range types,
/// negative indexing for accessing elements from the end of a dimension, and stepped
/// slicing using semicolon notation.
///
/// # Syntax
///
/// - `s![range]` - Single dimension slice
/// - `s![range1, range2, ...]` - Multi-dimensional slice
/// - `s![range;step]` - Slice with custom step
/// - `s![range1;step1, range2, range3;step3]` - Mixed regular and stepped slices
///
/// # Step Values
///
/// - Positive steps (e.g., `2`) select every nth element forward
/// - Negative steps (e.g., `-1`) reverse the selection within the range
/// - Step of `1` is the default when not specified
/// - Step cannot be `0`
///
/// # Negative Step Semantics
///
/// When using negative steps, the range still defines which elements to consider,
/// but iteration proceeds backwards from the end of that range:
/// - `s![0..5;-1]` selects indices [4, 3, 2, 1, 0]
/// - `s![2..8;-2]` selects indices [7, 5, 3]
///
/// # Examples
///
/// ```rust,ignore
/// // Basic slicing
/// let slice = tensor.slice(s![0..5, .., 3]);
///
/// // Using negative indices (counting from the end)
/// let last_row = tensor.slice(s![-1, ..]);
///
/// // Slicing with positive steps (every 2nd element)
/// let even = tensor.slice(s![0..10;2]);
///
/// // Reversing a dimension
/// let reversed = tensor.slice(s![0..5;-1]);
///
/// // Complex multi-dimensional slicing
/// let complex = tensor.slice(s![0..10;2, .., 5..15;-3, -1]);
///
/// // Mixed range types and steps
/// let mixed = tensor.slice(s![2..=5, 1..;2, ..10;-1]);
/// ```
#[macro_export]
macro_rules! s {
    // Empty - should not happen
    [] => {
        compile_error!("Empty slice specification")
    };

    // Single expression with step
    [$range:expr; $step:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::Slice::from_range_stepped($range, $step)
            }
        }
    };

    // Single expression without step (no comma after)
    [$range:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::Slice::from($range)
            }
        }
    };

    // Two or more expressions with first having step
    [$range:expr; $step:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::Slice::from_range_stepped($range, $step)] $($rest)*)
            }
        }
    };

    // Two or more expressions with first not having step
    [$range:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::Slice::from($range)] $($rest)*)
            }
        }
    };

    // Internal: finished parsing
    (@internal [$($acc:expr),*]) => {
        [$($acc),*]
    };

    // Internal: parse range with step followed by comma
    (@internal [$($acc:expr),*] $range:expr; $step:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::Slice::from_range_stepped($range, $step as isize)] $($rest)*)
    };

    // Internal: parse range with step at end
    (@internal [$($acc:expr),*] $range:expr; $step:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::Slice::from_range_stepped($range, $step as isize)])
    };

    // Internal: parse range without step followed by comma
    (@internal [$($acc:expr),*] $range:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::Slice::from($range)] $($rest)*)
    };

    // Internal: parse range without step at end
    (@internal [$($acc:expr),*] $range:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::Slice::from($range)])
    };
}

/// A slice (range) with optional step.
///
/// - `end` is an exclusive index.
/// - Negative `start` or `end` indices are counted from the back of the axis.
/// - If `end` is `None`, the slice extends to the end of the axis.
/// - `step` defines the stride between elements (default: 1).
/// - Negative `step` reverses the slice direction.
///
/// See also the [`s![]`](s!) macro.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Slice {
    /// Slice start index.
    pub start: isize,
    /// Slice end index (exclusive).
    pub end: Option<isize>,
    /// Step between elements (default: 1).
    pub step: isize,
}

impl Slice {
    /// Creates a new slice with start, end, and step
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { start, end, step }
    }

    /// Creates a slice that represents a single index
    pub fn index(idx: isize) -> Self {
        Self {
            start: idx,
            end: handle_signed_inclusive_end(idx),
            step: 1,
        }
    }

    /// Creates a slice with a custom step
    pub fn with_step(start: isize, end: Option<isize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { start, end, step }
    }

    /// Creates a slice from a range with a specified step
    pub fn from_range_stepped<R: Into<Slice>>(range: R, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        let mut slice = range.into();
        slice.step = step;
        slice
    }

    /// Returns the step of the slice
    pub fn step(&self) -> isize {
        self.step
    }

    /// Returns the range for this slice given a dimension size
    pub fn range(&self, size: usize) -> Range<usize> {
        self.to_range(size)
    }

    /// Convert this slice to a range for a dimension of the given size.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the dimension to slice.
    ///
    /// # Returns
    ///
    /// A `Range<usize>` representing the slice bounds.
    pub fn to_range(&self, size: usize) -> Range<usize> {
        // Always return a valid range with start <= end
        // The step information will be handled separately
        let start = convert_signed_index(self.start, size);
        let end = match self.end {
            Some(end) => convert_signed_index(end, size),
            None => size,
        };
        start..end
    }

    /// Converts the slice into a range and step tuple
    pub fn to_range_and_step(&self, size: usize) -> (Range<usize>, isize) {
        let range = self.to_range(size);
        (range, self.step)
    }

    /// Returns true if the step is negative
    pub fn is_reversed(&self) -> bool {
        self.step < 0
    }

    /// Calculates the output size for this slice operation
    pub fn output_size(&self, dim_size: usize) -> usize {
        let range = self.to_range(dim_size);
        let len = range.end - range.start;
        if self.step.unsigned_abs() == 1 {
            len
        } else {
            len.div_ceil(self.step.unsigned_abs())
        }
    }
}

fn convert_signed_index(index: isize, size: usize) -> usize {
    if index < 0 {
        (size as isize + index).max(0) as usize
    } else {
        (index as usize).min(size)
    }
}

fn handle_signed_inclusive_end(end: isize) -> Option<isize> {
    match end {
        -1 => None,
        end => Some(end + 1),
    }
}

impl<I: AsIndex> From<Range<I>> for Slice {
    fn from(r: Range<I>) -> Self {
        Self {
            start: r.start.index(),
            end: Some(r.end.index()),
            step: 1,
        }
    }
}

impl<I: AsIndex + Copy> From<RangeInclusive<I>> for Slice {
    fn from(r: RangeInclusive<I>) -> Self {
        Self {
            start: (*r.start()).index(),
            end: handle_signed_inclusive_end((*r.end()).index()),
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeFrom<I>> for Slice {
    fn from(r: RangeFrom<I>) -> Self {
        Self {
            start: r.start.index(),
            end: None,
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeTo<I>> for Slice {
    fn from(r: RangeTo<I>) -> Self {
        Self {
            start: 0,
            end: Some(r.end.index()),
            step: 1,
        }
    }
}

impl<I: AsIndex> From<RangeToInclusive<I>> for Slice {
    fn from(r: RangeToInclusive<I>) -> Self {
        Self {
            start: 0,
            end: handle_signed_inclusive_end(r.end.index()),
            step: 1,
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<usize> for Slice {
    fn from(i: usize) -> Self {
        Slice::index(i as isize)
    }
}

impl From<isize> for Slice {
    fn from(i: isize) -> Self {
        Slice::index(i)
    }
}

impl From<i32> for Slice {
    fn from(i: i32) -> Self {
        Slice::index(i as isize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_calculate_slice_output_shape_basic() {
        // Test basic slicing with step=1
        let slices = vec![
            Slice::new(0, Some(5), 1), // 5 elements
            Slice::new(2, Some(8), 1), // 6 elements
        ];
        let original_shape = vec![10, 10, 10];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![5, 6, 10]);
    }

    #[test]
    fn test_calculate_slice_output_shape_with_positive_steps() {
        // Test slicing with various positive steps
        let slices = vec![
            Slice::new(0, Some(10), 2), // [0,2,4,6,8] -> 5 elements
            Slice::new(1, Some(9), 3),  // [1,4,7] -> 3 elements
            Slice::new(0, Some(7), 4),  // [0,4] -> 2 elements
        ];
        let original_shape = vec![20, 20, 20, 30];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![5, 3, 2, 30]);
    }

    #[test]
    fn test_calculate_slice_output_shape_with_negative_steps() {
        // Test slicing with negative steps (backward iteration)
        let slices = vec![
            Slice::new(0, Some(10), -1), // 10 elements traversed backward
            Slice::new(2, Some(8), -2),  // [7,5,3] -> 3 elements
        ];
        let original_shape = vec![20, 20, 20];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![10, 3, 20]);
    }

    #[test]
    fn test_calculate_slice_output_shape_mixed_steps() {
        // Test with a mix of positive, negative, and unit steps
        let slices = vec![
            Slice::from_range_stepped(1..6, 1),   // 5 elements
            Slice::from_range_stepped(0..10, -3), // [9,6,3,0] -> 4 elements
            Slice::from_range_stepped(2..14, 4),  // [2,6,10] -> 3 elements
        ];
        let original_shape = vec![20, 20, 20];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![5, 4, 3]);
    }

    #[test]
    fn test_calculate_slice_output_shape_partial_dims() {
        // Test when slices has fewer dimensions than original shape
        let slices = vec![
            Slice::from_range_stepped(2..7, 2), // [2,4,6] -> 3 elements
        ];
        let original_shape = vec![10, 20, 30, 40];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![3, 20, 30, 40]);
    }

    #[test]
    fn test_calculate_slice_output_shape_edge_cases() {
        // Test edge cases with small ranges and large steps
        let slices = vec![
            Slice::from_range_stepped(0..1, 1),    // Single element
            Slice::from_range_stepped(0..10, 100), // Step larger than range -> 1 element
            Slice::from_range_stepped(5..5, 1),    // Empty range -> 0 elements
        ];
        let original_shape = vec![10, 20, 30];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![1, 1, 0]);
    }

    #[test]
    fn test_slice_output_size() {
        // Test the output_size method directly
        assert_eq!(Slice::new(0, Some(10), 1).output_size(10), 10);
        assert_eq!(Slice::new(0, Some(10), 2).output_size(10), 5);
        assert_eq!(Slice::new(0, Some(10), 3).output_size(10), 4); // ceil(10/3)
        assert_eq!(Slice::new(0, Some(10), -1).output_size(10), 10);
        assert_eq!(Slice::new(0, Some(10), -2).output_size(10), 5);
        assert_eq!(Slice::new(2, Some(8), -3).output_size(10), 2); // ceil(6/3)
        assert_eq!(Slice::new(5, Some(5), 1).output_size(10), 0); // empty range
    }

    #[test]
    fn test_calculate_slice_output_shape_empty() {
        // Test with no slice infos (should return original shape)
        let slices = vec![];
        let original_shape = vec![10, 20, 30];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_calculate_slice_output_shape_uneven_division() {
        // Test cases where range size doesn't divide evenly by step
        let slices = vec![
            Slice::from_range_stepped(0..7, 3), // ceil(7/3) = 3 elements: [0,3,6]
            Slice::from_range_stepped(0..11, 4), // ceil(11/4) = 3 elements: [0,4,8]
            Slice::from_range_stepped(1..10, 5), // ceil(9/5) = 2 elements: [1,6]
        ];
        let original_shape = vec![20, 20, 20];
        let result = calculate_slice_output_shape(&slices, &original_shape);
        assert_eq!(result, vec![3, 3, 2]);
    }
}
