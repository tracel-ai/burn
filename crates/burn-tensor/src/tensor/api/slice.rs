use crate::indexing::AsIndex;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Information about a slice operation including range and step.
#[derive(Clone, Debug, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SliceInfo {
    /// The range of indices to select
    pub range: Range<usize>,
    /// The step between selected elements (default: 1)
    pub step: isize,
}

impl SliceInfo {
    /// Creates a new SliceInfo with the given range and step
    pub fn new(range: Range<usize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { range, step }
    }

    /// Creates a SliceInfo from a range with default step of 1
    pub fn from_range(range: Range<usize>) -> Self {
        Self { range, step: 1 }
    }

    /// Returns true if the step is negative
    pub fn is_reversed(&self) -> bool {
        self.step < 0
    }
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
                $crate::SliceWithStep::from_range_and_step($range, $step)
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
                $crate::s!(@internal [$crate::SliceWithStep::from_range_and_step($range, $step)] $($rest)*)
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
        $crate::s!(@internal [$($acc,)* $crate::SliceWithStep::from_range_and_step($range, $step)] $($rest)*)
    };

    // Internal: parse range with step at end
    (@internal [$($acc:expr),*] $range:expr; $step:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::SliceWithStep::from_range_and_step($range, $step)])
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
#[derive(new, Clone, Debug)]
pub struct Slice {
    /// Slice start index.
    start: isize,
    /// Slice end index (exclusive).
    end: Option<isize>,
    /// Step between elements (default: 1).
    step: isize,
}

impl Slice {
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

    /// Returns the step of the slice
    pub fn step(&self) -> isize {
        self.step
    }

    pub(crate) fn to_range(&self, size: usize) -> Range<usize> {
        // Always return a valid range with start <= end
        // The step information will be handled separately
        let start = convert_signed_index(self.start, size);
        let end = match self.end {
            Some(end) => convert_signed_index(end, size),
            None => size,
        };
        start..end
    }

    /// Converts the slice into a SliceInfo with range and step information
    pub(crate) fn to_slice_info(&self, size: usize) -> SliceInfo {
        let range = self.to_range(size);
        SliceInfo::new(range, self.step)
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

/// Helper struct for creating slices with steps
pub struct SliceWithStep;

/// Trait for types that can be used as step values
pub trait StepValue {
    /// Convert the step value to isize
    fn to_isize(self) -> isize;
}

impl StepValue for isize {
    fn to_isize(self) -> isize {
        self
    }
}

impl StepValue for i32 {
    fn to_isize(self) -> isize {
        self as isize
    }
}

impl StepValue for i64 {
    fn to_isize(self) -> isize {
        self as isize
    }
}

impl StepValue for usize {
    fn to_isize(self) -> isize {
        self as isize
    }
}

impl SliceWithStep {
    /// Creates a Slice from a range and step
    pub fn from_range_and_step<R, S>(range: R, step: S) -> Slice
    where
        R: Into<Slice>,
        S: StepValue,
    {
        let mut slice = range.into();
        let step_val = step.to_isize();
        assert!(step_val != 0, "Step cannot be zero");

        // Simply set the step value - the backend will handle the actual stepping
        // For negative steps, the backend will iterate backwards
        slice.step = step_val;
        slice
    }
}
