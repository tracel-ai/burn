use crate::baselib::indexing::AsIndex;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use derive_new::new;

/// Creates a slice specification for tensor indexing operations.
///
/// This macro simplifies the creation of tensor slices by allowing various range types
/// to be used together in a concise way. It supports all standard Rust range types
/// as well as negative indexing for accessing elements from the end of a dimension.
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
/// // Mixed range types
/// let complex_slice = tensor.slice(s![2..5, .., 0..=3, -2..]);
/// ```
#[macro_export]
macro_rules! s {
    [$range:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::baselib::indexing::Slice::from($range)
            }
        }
    };

    [$($range:expr),+] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                [$($crate::baselib::indexing::Slice::from($range)),+]
            }
        }
    };
}

/// A slice (range).
///
/// - `end` is an exclusive index.
/// - Negative `start` or `end` indices are counted from the back of the axis.
/// - If `end` is `None`, the slice extends to the end of the axis.
///
/// See also the [`s![]`](s!) macro.
#[derive(new, Clone, Debug)]
pub struct Slice {
    /// Slice start index.
    start: isize,
    /// Slice end index (exclusive).
    end: Option<isize>,
}

impl Slice {
    /// Creates a slice that represents a single index
    pub fn index(idx: isize) -> Self {
        Self {
            start: idx,
            end: handle_signed_inclusive_end(idx),
        }
    }

    /// Converts the slice into a `Range<usize>`.
    pub fn into_range(self, size: usize) -> Range<usize> {
        let start = convert_signed_index(self.start, size);

        let end = match self.end {
            Some(end) => convert_signed_index(end, size),
            None => size,
        };

        start..end
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
        }
    }
}

impl<I: AsIndex + Copy> From<RangeInclusive<I>> for Slice {
    fn from(r: RangeInclusive<I>) -> Self {
        Self {
            start: (*r.start()).index(),
            end: handle_signed_inclusive_end((*r.end()).index()),
        }
    }
}

impl<I: AsIndex> From<RangeFrom<I>> for Slice {
    fn from(r: RangeFrom<I>) -> Self {
        Self {
            start: r.start.index(),
            end: None,
        }
    }
}

impl<I: AsIndex> From<RangeTo<I>> for Slice {
    fn from(r: RangeTo<I>) -> Self {
        Self {
            start: 0,
            end: Some(r.end.index()),
        }
    }
}

impl<I: AsIndex> From<RangeToInclusive<I>> for Slice {
    fn from(r: RangeToInclusive<I>) -> Self {
        Self {
            start: 0,
            end: handle_signed_inclusive_end(r.end.index()),
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self {
            start: 0,
            end: None,
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
