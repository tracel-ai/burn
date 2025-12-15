//! A module for indexing utility machinery.

use crate::errors::BoundsError;
#[allow(unused_imports)]
use alloc::format;
#[allow(unused_imports)]
use alloc::string::{String, ToString};
use core::fmt::Debug;

/// Helper trait for implementing indexing with support for negative indices.
///
/// # Example
/// ```rust
/// use burn_std::AsIndex;
///
/// fn example<I: AsIndex, const D: usize>(dim: I, size: usize) -> isize {
///    let dim: usize = dim.expect_dim(D);
///    unimplemented!()
/// }
/// ```
pub trait AsIndex: Debug + Copy + Sized {
    /// Converts into a slice index.
    fn index(self) -> isize;

    /// Short-form `NegativeWrap::expect_index(idx, size)`.
    fn expect_index(self, size: usize) -> usize {
        NegativeWrap::expect_index(self, size)
    }

    /// Short-form `NegativeWrap::expect_dim(idx, size)`.
    fn expect_dim(self, size: usize) -> usize {
        NegativeWrap::expect_dim(self, size)
    }
}

impl AsIndex for usize {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for isize {
    fn index(self) -> isize {
        self
    }
}

impl AsIndex for i64 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u64 {
    fn index(self) -> isize {
        self as isize
    }
}

// Default integer type
impl AsIndex for i32 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u32 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for i16 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u16 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for i8 {
    fn index(self) -> isize {
        self as isize
    }
}

impl AsIndex for u8 {
    fn index(self) -> isize {
        self as isize
    }
}

/// Wraps an index with negative indexing support.
#[derive(Debug)]
pub struct NegativeWrap {
    index_name: &'static str,
    size_name: &'static str,
    wrap_scalar: bool,
}

impl NegativeWrap {
    /// Get an instance for wrapping negative indices.
    pub fn index() -> Self {
        Self {
            index_name: "index",
            size_name: "size",
            wrap_scalar: false,
        }
    }

    /// Get an instance for wrapping negative dimensions.
    pub fn dim() -> Self {
        Self {
            index_name: "dimension index",
            size_name: "rank",
            wrap_scalar: false,
        }
    }

    /// Set the policy for wrapping 0-size ranges.
    ///
    /// - when size == 0:
    ///   - if wrap_scalar; then size == 1
    ///   - otherwise; an error.
    pub fn with_wrap_scalar(self, wrap_scalar: bool) -> Self {
        Self {
            wrap_scalar,
            ..self
        }
    }

    /// Wrap an index with negative indexing support.
    pub fn try_wrap<I: AsIndex>(&self, idx: I, size: usize) -> Result<usize, BoundsError> {
        try_wrap(idx, size, self.index_name, self.size_name, self.wrap_scalar)
    }

    /// Wrap an index with negative indexing support.
    pub fn expect_wrap<I: AsIndex>(&self, idx: I, size: usize) -> usize {
        expect_wrap(idx, size, self.index_name, self.size_name, self.wrap_scalar)
    }

    /// Short-form `NegativeWrap::index().expect_wrap(idx, size)`.
    pub fn expect_index<I: AsIndex>(idx: I, size: usize) -> usize {
        Self::index().expect_wrap(idx, size)
    }

    /// Short-form `NegativeWrap::dim().expect_wrap(idx, size)`.
    pub fn expect_dim<I: AsIndex>(idx: I, size: usize) -> usize {
        Self::dim().expect_wrap(idx, size)
    }
}

/// Wraps an index with negative indexing support.
///
/// ## Arguments
/// - `idx` - The index to canonicalize.
/// - `size` - The size of the index range.
/// - `index_name` - The name of the index (for error messages).
/// - `size_name` - The name of the size (for error messages).
/// - `wrap_scalar` - If true, treat 0-size ranges as having size 1.
///
/// ## Returns
///
/// A `Result<usize, BoundsError>` of the canonicalized index.
pub fn expect_wrap<I>(
    idx: I,
    size: usize,
    index_name: &str,
    size_name: &str,
    wrap_scalar: bool,
) -> usize
where
    I: AsIndex,
{
    try_wrap(idx, size, index_name, size_name, wrap_scalar).expect("valid index")
}

/// Wraps an index with negative indexing support.
///
/// ## Arguments
/// - `idx` - The index to canonicalize.
/// - `size` - The size of the index range.
/// - `index_name` - The name of the index (for error messages).
/// - `size_name` - The name of the size (for error messages).
/// - `wrap_scalar` - If true, treat 0-size ranges as having size 1.
///
/// ## Returns
///
/// A `Result<usize, BoundsError>` of the canonicalized index.
pub fn try_wrap<I>(
    idx: I,
    size: usize,
    index_name: &str,
    size_name: &str,
    wrap_scalar: bool,
) -> Result<usize, BoundsError>
where
    I: AsIndex,
{
    let idx = idx.index();

    let _size = if size > 0 {
        size
    } else {
        if !wrap_scalar {
            return Err(BoundsError {
                index_name: index_name.to_string(),
                index: idx.to_string(),
                bounds_name: size_name.to_string(),
                bounds: size.to_string(),
            });
        }
        1
    };

    if idx >= 0 && (idx as usize) < _size {
        return Ok(idx as usize);
    }

    let _idx = if idx < 0 { idx + _size as isize } else { idx };

    if _idx < 0 || (_idx as usize) >= _size {
        let rank = _size as isize;
        let upper = rank - 1;

        return Err(BoundsError {
            index_name: index_name.to_string(),
            index: idx.to_string(),
            bounds_name: size_name.to_string(),
            bounds: format!("0..={upper}"),
        });
    }

    Ok(_idx as usize)
}

/// Wraps a dimension index to be within the bounds of the dimension size.
///
/// ## Arguments
///
/// * `idx` - The dimension index to wrap.
/// * `size` - The size of the dimension.
///
/// ## Returns
///
/// The positive wrapped dimension index.
#[inline]
#[must_use]
pub fn wrap_index<I>(idx: I, size: usize) -> usize
where
    I: AsIndex,
{
    if size == 0 {
        return 0; // Avoid modulo by zero
    }
    let wrapped = idx.index().rem_euclid(size as isize);
    if wrapped < 0 {
        (wrapped + size as isize) as usize
    } else {
        wrapped as usize
    }
}

/// Compute the ravel index for the given coordinates.
///
/// This returns the row-major order raveling:
/// * `strides[-1] = 1`
/// * `strides[i] = strides[i+1] * dims[i+1]`
/// * `dim_strides = coords * strides`
/// * `ravel = sum(dim_strides)`
///
/// # Arguments
/// - `indices`: the index for each dimension; must be the same length as `shape`.
/// - `shape`: the shape of each dimension; be the same length as `indices`.
///
/// # Returns
/// - the ravel offset index.
pub fn ravel_index<I: AsIndex>(indices: &[I], shape: &[usize]) -> usize {
    assert_eq!(
        shape.len(),
        indices.len(),
        "Coordinate rank mismatch: expected {}, got {}",
        shape.len(),
        indices.len(),
    );

    let mut ravel_idx = 0;
    let mut stride = 1;

    for (i, &dim) in shape.iter().enumerate().rev() {
        let idx = indices[i];
        let coord = NegativeWrap::index().expect_wrap(idx, dim);
        ravel_idx += coord * stride;
        stride *= dim;
    }

    ravel_idx
}

#[cfg(test)]
#[allow(clippy::identity_op, reason = "useful for clarity")]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_ravel() {
        let shape = vec![2, 3, 4, 5];

        assert_eq!(ravel_index(&[0, 0, 0, 0], &shape), 0);
        assert_eq!(
            ravel_index(&[1, 2, 3, 4], &shape),
            1 * (3 * 4 * 5) + 2 * (4 * 5) + 3 * 5 + 4
        );
    }

    #[test]
    fn test_wrap_idx() {
        assert_eq!(wrap_index(0, 3), 0_usize);
        assert_eq!(wrap_index(3, 3), 0_usize);
        assert_eq!(wrap_index(2 * 3, 3), 0_usize);
        assert_eq!(wrap_index(0 - 3, 3), 0_usize);
        assert_eq!(wrap_index(0 - 2 * 3, 3), 0_usize);

        assert_eq!(wrap_index(1, 3), 1_usize);
        assert_eq!(wrap_index(1 + 3, 3), 1_usize);
        assert_eq!(wrap_index(1 + 2 * 3, 3), 1_usize);
        assert_eq!(wrap_index(1 - 3, 3), 1_usize);
        assert_eq!(wrap_index(1 - 2 * 3, 3), 1_usize);

        assert_eq!(wrap_index(2, 3), 2_usize);
        assert_eq!(wrap_index(2 + 3, 3), 2_usize);
        assert_eq!(wrap_index(2 + 2 * 3, 3), 2_usize);
        assert_eq!(wrap_index(2 - 3, 3), 2_usize);
        assert_eq!(wrap_index(2 - 2 * 3, 3), 2_usize);
    }

    #[test]
    fn test_negative_wrap() {
        assert_eq!(NegativeWrap::index().expect_wrap(0, 3), 0);
        assert_eq!(NegativeWrap::index().expect_wrap(1, 3), 1);
        assert_eq!(NegativeWrap::index().expect_wrap(2, 3), 2);
        assert_eq!(NegativeWrap::index().expect_wrap(-1, 3), 2);
        assert_eq!(NegativeWrap::index().expect_wrap(-2, 3), 1);
        assert_eq!(NegativeWrap::index().expect_wrap(-3, 3), 0);

        assert_eq!(NegativeWrap::dim().expect_wrap(0, 3), 0);
        assert_eq!(NegativeWrap::dim().expect_wrap(1, 3), 1);
        assert_eq!(NegativeWrap::dim().expect_wrap(2, 3), 2);
        assert_eq!(NegativeWrap::dim().expect_wrap(-1, 3), 2);
        assert_eq!(NegativeWrap::dim().expect_wrap(-2, 3), 1);
        assert_eq!(NegativeWrap::dim().expect_wrap(-3, 3), 0);

        assert_eq!(
            NegativeWrap::index().try_wrap(3, 3),
            Err(BoundsError {
                index_name: "index".to_string(),
                index: "3".to_string(),
                bounds_name: "size".to_string(),
                bounds: "0..=2".to_string()
            })
        );
        assert_eq!(
            NegativeWrap::index().try_wrap(-4, 3),
            Err(BoundsError {
                index_name: "index".to_string(),
                index: "-4".to_string(),
                bounds_name: "size".to_string(),
                bounds: "0..=2".to_string()
            })
        );
        assert_eq!(
            NegativeWrap::dim().try_wrap(3, 3),
            Err(BoundsError {
                index_name: "dimension index".to_string(),
                index: "3".to_string(),
                bounds_name: "rank".to_string(),
                bounds: "0..=2".to_string()
            })
        );
        assert_eq!(
            NegativeWrap::dim().try_wrap(-4, 3),
            Err(BoundsError {
                index_name: "dimension index".to_string(),
                index: "-4".to_string(),
                bounds_name: "rank".to_string(),
                bounds: "0..=2".to_string()
            })
        );
    }

    #[test]
    fn test_negative_wrap_scalar() {
        assert_eq!(
            NegativeWrap::index().try_wrap(0, 0),
            Err(BoundsError {
                index_name: "index".to_string(),
                index: "0".to_string(),
                bounds_name: "size".to_string(),
                bounds: "0".to_string()
            })
        );

        assert_eq!(
            NegativeWrap::index()
                .with_wrap_scalar(true)
                .expect_wrap(0, 0),
            0
        );
        assert_eq!(
            NegativeWrap::index()
                .with_wrap_scalar(true)
                .expect_wrap(-1, 0),
            0
        );

        assert_eq!(
            NegativeWrap::index().with_wrap_scalar(false).try_wrap(1, 0),
            Err(BoundsError {
                index_name: "index".to_string(),
                index: "1".to_string(),
                bounds_name: "size".to_string(),
                bounds: "0".to_string()
            })
        );
    }
}
