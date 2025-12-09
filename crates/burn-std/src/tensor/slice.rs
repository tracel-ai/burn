//! Tensor slice utilities.

use crate::Shape;
use crate::indexing::AsIndex;
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use core::str::FromStr;

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
                $crate::tensor::Slice::from_range_stepped($range, $step)
            }
        }
    };

    // Single expression without step (no comma after)
    [$range:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::tensor::Slice::from($range)
            }
        }
    };

    // Two or more expressions with first having step
    [$range:expr; $step:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::tensor::Slice::from_range_stepped($range, $step)] $($rest)*)
            }
        }
    };

    // Two or more expressions with first not having step
    [$range:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::tensor::Slice::from($range)] $($rest)*)
            }
        }
    };

    // Internal: finished parsing
    (@internal [$($acc:expr),*]) => {
        [$($acc),*]
    };

    // Internal: parse range with step followed by comma
    (@internal [$($acc:expr),*] $range:expr; $step:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from_range_stepped($range, $step as isize)] $($rest)*)
    };

    // Internal: parse range with step at end
    (@internal [$($acc:expr),*] $range:expr; $step:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from_range_stepped($range, $step as isize)])
    };

    // Internal: parse range without step followed by comma
    (@internal [$($acc:expr),*] $range:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from($range)] $($rest)*)
    };

    // Internal: parse range without step at end
    (@internal [$($acc:expr),*] $range:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from($range)])
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

/// Defines an [`Iterator`] over a [`Slice`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SliceIter {
    slice: Slice,
    current: isize,
}

impl Iterator for SliceIter {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.current;
        self.current += self.slice.step;

        if let Some(end) = self.slice.end {
            if self.slice.is_reversed() {
                if next <= end {
                    return None;
                }
            } else if next >= end {
                return None;
            }
        }

        Some(next)
    }
}

/// Note: Unbounded [`Slice`]s produce infinite iterators.
impl IntoIterator for Slice {
    type Item = isize;
    type IntoIter = SliceIter;

    fn into_iter(self) -> Self::IntoIter {
        SliceIter {
            slice: self,
            current: self.start,
        }
    }
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

    /// Converts the slice to a vector.
    pub fn into_vec(self) -> Vec<isize> {
        assert!(
            self.end.is_some(),
            "Slice must have an end to convert to a vector: {self:?}"
        );
        self.into_iter().collect()
    }

    /// Clips the slice to a maximum size.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert_eq!(
    ///     Slice::new(0, None, 1).bound_to(10),
    ///     Slice::new(0, Some(10), 1));
    /// assert_eq!(
    ///     Slice::new(0, Some(5), 1).bound_to(10),
    ///     Slice::new(0, Some(5), 1));
    /// assert_eq!(
    ///     Slice::new(0, None, -1).bound_to(10),
    ///     Slice::new(0, Some(-11), -1));
    /// assert_eq!(
    ///     Slice::new(0, Some(-5), -1).bound_to(10),
    ///     Slice::new(0, Some(-5), -1));
    /// ```
    pub fn bound_to(self, size: usize) -> Self {
        let mut bounds = size as isize;

        if let Some(end) = self.end {
            if end > 0 {
                bounds = end.min(bounds);
            } else {
                bounds = end.max(-(bounds + 1));
            }
        } else if self.is_reversed() {
            bounds = -(bounds + 1);
        }

        Self {
            end: Some(bounds),
            ..self
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
        // Handle empty slices (start >= end)
        if range.start >= range.end {
            return 0;
        }
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

impl Display for Slice {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        if self.step == 1
            && let Some(end) = self.end
            && self.start == end - 1
        {
            f.write_fmt(format_args!("{}", self.start))
        } else {
            if self.start != 0 {
                f.write_fmt(format_args!("{}", self.start))?;
            }
            f.write_str("..")?;
            if let Some(end) = self.end {
                f.write_fmt(format_args!("{}", end))?;
            }
            if self.step != 1 {
                f.write_fmt(format_args!(";{}", self.step))?;
            }
            Ok(())
        }
    }
}

impl FromStr for Slice {
    type Err = SliceExpressionError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut s = source.trim();

        let parse_int = |v: &str| -> Result<isize, Self::Err> {
            v.parse::<isize>().map_err(|e| {
                SliceExpressionError::parse_error(format!("Invalid integer: '{v}': {}", e), source)
            })
        };

        let mut start: isize = 0;
        let mut end: Option<isize> = None;
        let mut step: isize = 1;

        if let Some((head, tail)) = s.split_once(";") {
            step = parse_int(tail)?;
            s = head;
        }

        if s.is_empty() {
            return Err(SliceExpressionError::parse_error(
                "Empty expression",
                source,
            ));
        }

        if let Some((start_s, end_s)) = s.split_once("..") {
            if !start_s.is_empty() {
                start = parse_int(start_s)?;
            }
            if !end_s.is_empty() {
                if let Some(end_s) = end_s.strip_prefix('=') {
                    end = Some(parse_int(end_s)? + 1);
                } else {
                    end = Some(parse_int(end_s)?);
                }
            }
        } else {
            start = parse_int(s)?;
            end = Some(start + 1);
        }

        if step == 0 {
            return Err(SliceExpressionError::invalid_expression(
                "Step cannot be zero",
                source,
            ));
        }

        Ok(Slice::new(start, end, step))
    }
}

#[allow(unused_imports)]
use alloc::format;
use alloc::string::String;

/// Common Parse Error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceExpressionError {
    /// Parse Error.
    ParseError {
        /// The error message.
        message: String,
        /// The source expression.
        source: String,
    },

    /// Invalid Expression.
    InvalidExpression {
        /// The error message.
        message: String,
        /// The source expression.
        source: String,
    },
}

impl SliceExpressionError {
    /// Constructs a new `ParseError`.
    ///
    /// This function is a utility for creating instances where a parsing error needs to be represented,
    /// encapsulating a descriptive error message and the source of the error.
    ///
    /// # Parameters
    ///
    /// - `message`: A value that can be converted into a `String`, representing a human-readable description
    ///   of the parsing error.
    /// - `source`: A value that can be converted into a `String`, typically identifying the origin or
    ///   input that caused the parsing error.
    pub fn parse_error(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
            source: source.into(),
        }
    }

    /// Creates a new `InvalidExpression`.
    ///
    /// # Parameters
    /// - `message`: A detailed message describing the nature of the invalid expression.
    ///   Accepts any type that can be converted into a `String`.
    /// - `source`: The source or context in which the invalid expression occurred.
    ///   Accepts any type that can be converted into a `String`.
    pub fn invalid_expression(message: impl Into<String>, source: impl Into<String>) -> Self {
        Self::InvalidExpression {
            message: message.into(),
            source: source.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn test_parse_error() {
        let err = SliceExpressionError::parse_error("test", "source");
        assert_eq!(
            format!("{:?}", err),
            "ParseError { message: \"test\", source: \"source\" }"
        );
    }

    #[test]
    fn test_invalid_expression() {
        let err = SliceExpressionError::invalid_expression("test", "source");
        assert_eq!(
            format!("{:?}", err),
            "InvalidExpression { message: \"test\", source: \"source\" }"
        );
    }

    #[test]
    fn test_slice_to_str() {
        assert_eq!(Slice::new(0, None, 1).to_string(), "..");

        assert_eq!(Slice::new(0, Some(1), 1).to_string(), "0");

        assert_eq!(Slice::new(0, Some(10), 1).to_string(), "..10");
        assert_eq!(Slice::new(1, Some(10), 1).to_string(), "1..10");

        assert_eq!(Slice::new(-3, Some(10), -2).to_string(), "-3..10;-2");
    }

    #[test]
    fn test_slice_from_str() {
        assert_eq!("1".parse::<Slice>(), Ok(Slice::new(1, Some(2), 1)));
        assert_eq!("..".parse::<Slice>(), Ok(Slice::new(0, None, 1)));
        assert_eq!("..3".parse::<Slice>(), Ok(Slice::new(0, Some(3), 1)));
        assert_eq!("..=3".parse::<Slice>(), Ok(Slice::new(0, Some(4), 1)));

        assert_eq!("-12..3".parse::<Slice>(), Ok(Slice::new(-12, Some(3), 1)));
        assert_eq!("..;-1".parse::<Slice>(), Ok(Slice::new(0, None, -1)));

        assert_eq!("..=3;-2".parse::<Slice>(), Ok(Slice::new(0, Some(4), -2)));

        assert_eq!(
            "..;0".parse::<Slice>(),
            Err(SliceExpressionError::invalid_expression(
                "Step cannot be zero",
                "..;0"
            ))
        );

        assert_eq!(
            "".parse::<Slice>(),
            Err(SliceExpressionError::parse_error("Empty expression", ""))
        );
        assert_eq!(
            "a".parse::<Slice>(),
            Err(SliceExpressionError::parse_error(
                "Invalid integer: 'a': invalid digit found in string",
                "a"
            ))
        );
        assert_eq!(
            "..a".parse::<Slice>(),
            Err(SliceExpressionError::parse_error(
                "Invalid integer: 'a': invalid digit found in string",
                "..a"
            ))
        );
        assert_eq!(
            "a:b:c".parse::<Slice>(),
            Err(SliceExpressionError::parse_error(
                "Invalid integer: 'a:b:c': invalid digit found in string",
                "a:b:c"
            ))
        );
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
    fn test_bound_to() {
        assert_eq!(
            Slice::new(0, None, 1).bound_to(10),
            Slice::new(0, Some(10), 1)
        );
        assert_eq!(
            Slice::new(0, Some(5), 1).bound_to(10),
            Slice::new(0, Some(5), 1)
        );

        assert_eq!(
            Slice::new(0, None, -1).bound_to(10),
            Slice::new(0, Some(-11), -1)
        );
        assert_eq!(
            Slice::new(0, Some(-5), -1).bound_to(10),
            Slice::new(0, Some(-5), -1)
        );
    }

    #[test]
    fn test_slice_iter() {
        assert_eq!(
            Slice::new(2, Some(3), 1).into_iter().collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            Slice::new(3, Some(-1), -1).into_iter().collect::<Vec<_>>(),
            vec![3, 2, 1, 0]
        );

        assert_eq!(Slice::new(3, Some(-1), -1).into_vec(), vec![3, 2, 1, 0]);

        assert_eq!(
            Slice::new(3, None, 2)
                .into_iter()
                .take(3)
                .collect::<Vec<_>>(),
            vec![3, 5, 7]
        );
        assert_eq!(
            Slice::new(3, None, 2)
                .bound_to(8)
                .into_iter()
                .collect::<Vec<_>>(),
            vec![3, 5, 7]
        );
    }

    #[test]
    #[should_panic(
        expected = "Slice must have an end to convert to a vector: Slice { start: 0, end: None, step: 1 }"
    )]
    fn test_unbound_slice_into_vec() {
        Slice::new(0, None, 1).into_vec();
    }
}
