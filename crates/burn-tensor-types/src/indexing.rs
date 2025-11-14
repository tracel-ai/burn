//! A module for indexing utility machinery.

use core::fmt::Debug;

use crate::Element;

/// Helper trait for implementing indexing with support for negative indices.
///
/// # Example
/// ```rust
/// use burn_tensor::indexing::{AsIndex, canonicalize_dim};
///
/// fn example<I: AsIndex, const D: usize>(dim: I, size: usize) -> isize {
///    let dim: usize = canonicalize_dim(dim, D, false);
///    unimplemented!()
/// }
/// ```
pub trait AsIndex: Debug + Copy + Sized {
    /// Converts into a slice index.
    fn index(self) -> isize;
}

// TODO: split `Element` for int/float elems
impl<E: Element> AsIndex for E {
    fn index(self) -> isize {
        self.to_isize()
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

/// Canonicalizes and bounds checks an index with negative indexing support.
///
/// ## Arguments
///
/// * `idx` - The index to canonicalize.
/// * `size` - The size of the index range.
/// * `wrap_scalar` - If true, pretend scalars have rank=1.
///
/// ## Returns
///
/// The canonicalized dimension index.
///
/// ## Panics
///
/// * If `wrap_scalar` is false and the tensor has no dimensions.
/// * If the dimension index is out of range.
#[must_use]
pub fn canonicalize_index<Index>(idx: Index, size: usize, wrap_scalar: bool) -> usize
where
    Index: AsIndex,
{
    canonicalize_named_index("index", "size", idx, size, wrap_scalar)
}

/// Canonicalizes and bounds checks a dimension index with negative indexing support.
///
/// ## Arguments
///
/// * `idx` - The dimension index to canonicalize.
/// * `rank` - The number of dimensions.
/// * `wrap_scalar` - If true, pretend scalars have rank=1.
///
/// ## Returns
///
/// The canonicalized dimension index.
///
/// ## Panics
///
/// * If `wrap_scalar` is false and the tensor has no dimensions.
/// * If the dimension index is out of range.
#[must_use]
pub fn canonicalize_dim<Dim>(idx: Dim, rank: usize, wrap_scalar: bool) -> usize
where
    Dim: AsIndex,
{
    canonicalize_named_index("dimension index", "rank", idx, rank, wrap_scalar)
}

/// Canonicalizes and bounds checks an index with negative indexing support.
///
/// ## Arguments
///
/// * `name` - The name of the index (for error messages).
/// * `size_name` - The name of the size (for error messages).
/// * `idx` - The index to canonicalize.
/// * `size` - The size of the index range.
/// * `wrap_scalar` - If true, treat 0-size ranges as having size 1.
///
/// ## Returns
///
/// The canonicalized index.
///
/// ## Panics
///
/// * If `wrap_scalar` is false and the size is 0.
/// * If the index is out of range for the dimension size.
#[inline(always)]
#[must_use]
fn canonicalize_named_index<I>(
    name: &str,
    size_name: &str,
    idx: I,
    size: usize,
    wrap_scalar: bool,
) -> usize
where
    I: AsIndex,
{
    let idx = idx.index();

    let rank = if size > 0 {
        size
    } else {
        if !wrap_scalar {
            panic!("{name} {idx} used when {size_name} is 0");
        }
        1
    };

    if idx >= 0 && (idx as usize) < rank {
        return idx as usize;
    }

    let _idx = if idx < 0 { idx + rank as isize } else { idx };

    if _idx < 0 || (_idx as usize) >= rank {
        let rank = rank as isize;
        let lower = -rank;
        let upper = rank - 1;
        panic!("{name} {idx} out of range: ({lower}..={upper})");
    }

    _idx as usize
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
        let coord = canonicalize_index(indices[i], dim, false);
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
    fn test_canonicalize_dim() {
        let wrap_scalar = false;
        assert_eq!(canonicalize_dim(0, 3, wrap_scalar), 0_usize);
        assert_eq!(canonicalize_dim(1, 3, wrap_scalar), 1_usize);
        assert_eq!(canonicalize_dim(2, 3, wrap_scalar), 2_usize);

        assert_eq!(canonicalize_dim(-1, 3, wrap_scalar), (3 - 1) as usize);
        assert_eq!(canonicalize_dim(-2, 3, wrap_scalar), (3 - 2) as usize);
        assert_eq!(canonicalize_dim(-3, 3, wrap_scalar), (3 - 3) as usize);

        let wrap_scalar = true;
        assert_eq!(canonicalize_dim(0, 0, wrap_scalar), 0);
        assert_eq!(canonicalize_dim(-1, 0, wrap_scalar), 0);
    }

    #[test]
    #[should_panic = "dimension index 0 used when rank is 0"]
    fn test_canonicalize_dim_error_no_dims() {
        let _d = canonicalize_dim(0, 0, false);
    }

    #[test]
    #[should_panic = "dimension index 3 out of range: (-3..=2)"]
    fn test_canonicalize_dim_error_too_big() {
        let _d = canonicalize_dim(3, 3, false);
    }
    #[test]
    #[should_panic = "dimension index -4 out of range: (-3..=2)"]
    fn test_canonicalize_dim_error_too_small() {
        let _d = canonicalize_dim(-4, 3, false);
    }

    #[test]
    fn test_canonicalize_index() {
        let wrap_scalar = false;
        assert_eq!(canonicalize_index(0, 3, wrap_scalar), 0_usize);
        assert_eq!(canonicalize_index(1, 3, wrap_scalar), 1_usize);
        assert_eq!(canonicalize_index(2, 3, wrap_scalar), 2_usize);

        assert_eq!(canonicalize_index(-1, 3, wrap_scalar), (3 - 1) as usize);
        assert_eq!(canonicalize_index(-2, 3, wrap_scalar), (3 - 2) as usize);
        assert_eq!(canonicalize_index(-3, 3, wrap_scalar), (3 - 3) as usize);

        let wrap_scalar = true;
        assert_eq!(canonicalize_index(0, 0, wrap_scalar), 0);
        assert_eq!(canonicalize_index(-1, 0, wrap_scalar), 0);
    }

    #[test]
    #[should_panic = "index 3 out of range: (-3..=2)"]
    fn test_canonicalize_index_error_too_big() {
        let _d = canonicalize_index(3, 3, false);
    }
}
