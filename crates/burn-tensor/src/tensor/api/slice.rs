use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

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
                $crate::Slice::from($range)
            }
        }
    };

    [$($range:expr),+] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                [$($crate::Slice::from($range)),+]
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

    pub(crate) fn into_range(self, size: usize) -> Range<usize> {
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

/// A helper trait to convert difference indices type to a slice index.
pub trait IndexConversion {
    /// Converts into a slice index.
    fn index(self) -> isize;
}

impl IndexConversion for usize {
    fn index(self) -> isize {
        self as isize
    }
}

impl IndexConversion for isize {
    fn index(self) -> isize {
        self
    }
}

// Default integer type
impl IndexConversion for i32 {
    fn index(self) -> isize {
        self as isize
    }
}

impl<I: IndexConversion> From<Range<I>> for Slice {
    fn from(r: Range<I>) -> Self {
        Self {
            start: r.start.index(),
            end: Some(r.end.index()),
        }
    }
}

impl<I: IndexConversion + Copy> From<RangeInclusive<I>> for Slice {
    fn from(r: RangeInclusive<I>) -> Self {
        Self {
            start: (*r.start()).index(),
            end: handle_signed_inclusive_end((*r.end()).index()),
        }
    }
}

impl<I: IndexConversion> From<RangeFrom<I>> for Slice {
    fn from(r: RangeFrom<I>) -> Self {
        Self {
            start: r.start.index(),
            end: None,
        }
    }
}

impl<I: IndexConversion> From<RangeTo<I>> for Slice {
    fn from(r: RangeTo<I>) -> Self {
        Self {
            start: 0,
            end: Some(r.end.index()),
        }
    }
}

impl<I: IndexConversion> From<RangeToInclusive<I>> for Slice {
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
