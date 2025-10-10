use alloc::vec::Vec;

use crate::Shape;
use crate::indexing::AsIndex;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Trait for slice arguments that can be converted into an array of slices.
/// This allows the `slice` method to accept both single slices (from `s![..]`)
/// and arrays of slices (from `s![.., ..]` or `[0..5, 1..3]`).
pub trait SliceArg<const D2: usize> {
    /// Convert to an array of slices with clamping to shape dimensions
    fn into_slices(self, shape: Shape) -> [Slice; D2];
}

impl<const D2: usize, T> SliceArg<D2> for [T; D2]
where
    T: Into<Slice>,
{
    fn into_slices(self, shape: Shape) -> [Slice; D2] {
        self.into_iter()
            .enumerate()
            .map(|(i, s)| {
                let slice: Slice = s.into();
                // Apply shape clamping by converting to range and back
                let clamped_range = slice.to_range(shape[i]);
                Slice::new(
                    clamped_range.start as isize,
                    Some(clamped_range.end as isize),
                    slice.step(),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

impl<T> SliceArg<1> for T
where
    T: Into<Slice>,
{
    fn into_slices(self, shape: Shape) -> [Slice; 1] {
        let slice: Slice = self.into();
        let clamped_range = slice.to_range(shape[0]);
        [Slice::new(
            clamped_range.start as isize,
            Some(clamped_range.end as isize),
            slice.step(),
        )]
    }
}

/// Slice argument constructor for tensor indexing.
///
/// The `s![]` macro is used to create multi-dimensional slice specifications for tensors.
/// It converts various range syntax forms into a `&[Slice]` that can be used with
/// `tensor.slice()` and `tensor.slice_assign()` operations.
///
/// # Syntax Overview
///
/// ## Basic Forms
///
/// * **`s![index]`** - Index a single element (produces a subview with that axis removed)
/// * **`s![range]`** - Slice a range of elements
/// * **`s![range;step]`** - Slice a range with a custom step
/// * **`s![dim1, dim2, ...]`** - Multiple dimensions, each can be any of the above forms
///
/// ## Range Types
///
/// All standard Rust range types are supported:
/// * **`a..b`** - From `a` (inclusive) to `b` (exclusive)
/// * **`a..=b`** - From `a` to `b` (both inclusive)
/// * **`a..`** - From `a` to the end
/// * **`..b`** - From the beginning to `b` (exclusive)
/// * **`..=b`** - From the beginning to `b` (inclusive)
/// * **`..`** - The full range (all elements)
///
/// ## Negative Indices
///
/// Negative indices count from the end of the axis:
/// * **`-1`** refers to the last element
/// * **`-2`** refers to the second-to-last element
/// * And so on...
///
/// This works in all range forms: `s![-3..-1]`, `s![-2..]`, `s![..-1]`
///
/// ## Step Syntax
///
/// Steps control the stride between selected elements:
/// * **`;step`** after a range specifies the step
/// * **Positive steps** select every nth element going forward
/// * **Negative steps** select every nth element going backward
/// * Default step is `1` when not specified
/// * Step cannot be `0`
///
/// ### Negative Step Behavior
///
/// With negative steps, the range bounds still specify *which* elements to include,
/// but the traversal order is reversed:
///
/// * `s![0..5;-1]` selects indices `[4, 3, 2, 1, 0]` (not `[0, 1, 2, 3, 4]`)
/// * `s![2..8;-2]` selects indices `[7, 5, 3]` (starting from 7, going backward by 2)
/// * `s![..;-1]` reverses the entire axis
///
/// This matches the semantics of NumPy and the ndarray crate.
///
/// # Examples
///
/// ## Basic Slicing
///
/// ```rust,ignore
/// use burn_tensor::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 3>) {
/// // Select rows 0-5 (exclusive)
/// let subset = tensor.slice(s![0..5, .., ..]);
///
/// // Select the last row
/// let last_row = tensor.slice(s![-1, .., ..]);
///
/// // Select columns 2, 3, 4
/// let cols = tensor.slice(s![.., 2..5, ..]);
///
/// // Select a single element at position [1, 2, 3]
/// let element = tensor.slice(s![1, 2, 3]);
/// # }
/// ```
///
/// ## Slicing with Steps
///
/// ```rust,ignore
/// use burn_tensor::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>) {
/// // Select every 2nd row
/// let even_rows = tensor.slice(s![0..10;2, ..]);
///
/// // Select every 3rd column
/// let cols = tensor.slice(s![.., 0..9;3]);
///
/// // Select every 2nd element in reverse order
/// let reversed_even = tensor.slice(s![10..0;-2, ..]);
/// # }
/// ```
///
/// ## Reversing Dimensions
///
/// ```rust,ignore
/// use burn_tensor::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>) {
/// // Reverse the first dimension
/// let reversed = tensor.slice(s![..;-1, ..]);
///
/// // Reverse both dimensions
/// let fully_reversed = tensor.slice(s![..;-1, ..;-1]);
///
/// // Reverse a specific range
/// let range_reversed = tensor.slice(s![2..8;-1, ..]);
/// # }
/// ```
///
/// ## Complex Multi-dimensional Slicing
///
/// ```rust,ignore
/// use burn_tensor::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 4>) {
/// // Mix of different slice types
/// let complex = tensor.slice(s![
///     0..10;2,    // Every 2nd element from 0 to 10
///     ..,         // All elements in dimension 1
///     5..15;-3,   // Every 3rd element from 14 down to 5
///     -1          // Last element in dimension 3
/// ]);
///
/// // Using inclusive ranges
/// let inclusive = tensor.slice(s![2..=5, 1..=3, .., ..]);
///
/// // Negative indices with steps
/// let from_end = tensor.slice(s![-5..-1;2, .., .., ..]);
/// # }
/// ```
///
/// ## Slice Assignment
///
/// ```rust,ignore
/// use burn_tensor::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>, values: Tensor<B, 2>) {
/// // Assign to every 2nd row
/// let tensor = tensor.slice_assign(s![0..10;2, ..], values);
///
/// // Assign to a reversed slice
/// let tensor = tensor.slice_assign(s![..;-1, 0..5], values);
/// # }
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

/// A slice specification for a single tensor dimension.
///
/// This struct represents a range with an optional step, used for advanced indexing
/// operations on tensors. It is typically created using the [`s!`] macro rather than
/// constructed directly.
///
/// # Fields
///
/// * `start` - The starting index (inclusive). Negative values count from the end.
/// * `end` - The ending index (exclusive). `None` means to the end of the dimension.
/// * `step` - The stride between elements. Must be non-zero.
///
/// # Index Interpretation
///
/// - **Positive indices**: Count from the beginning (0-based)
/// - **Negative indices**: Count from the end (-1 is the last element)
/// - **Bounds checking**: Indices are clamped to valid ranges
///
/// # Step Behavior
///
/// - **Positive step**: Traverse forward through the range
/// - **Negative step**: Traverse backward through the range
/// - **Step size**: Determines how many elements to skip
///
/// # Examples
///
/// While you typically use the [`s!`] macro, you can also construct slices directly:
///
/// ```rust,ignore
/// use burn_tensor::Slice;
///
/// // Equivalent to s![2..8]
/// let slice1 = Slice::new(2, Some(8), 1);
///
/// // Equivalent to s![0..10;2]
/// let slice2 = Slice::new(0, Some(10), 2);
///
/// // Equivalent to s![..;-1] (reverse)
/// let slice3 = Slice::new(0, None, -1);
/// ```
///
/// See also the [`s!`] macro for the preferred way to create slices.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Slice {
    /// Slice start index.
    pub start: isize,
    /// Slice end index (exclusive).
    pub end: Option<isize>,
    /// Step between elements (default: 1).
    pub step: isize,
}

impl Default for Slice {
    fn default() -> Self {
        Self::full()
    }
}

impl Slice {
    /// Creates a new slice with start, end, and step
    pub const fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { start, end, step }
    }

    /// Creates a slice that represents the full range.
    pub const fn full() -> Self {
        Self::new(0, None, 1)
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
}
