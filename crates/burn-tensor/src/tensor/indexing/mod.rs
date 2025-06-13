//! A module for dimension indexing utility machinery.

use core::fmt::Debug;

/// A helper trait to convert difference indices type to a slice index.
pub trait IndexConversion: Debug + Copy + Sized {
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

/// Canonicalizes and bounds checks a dimension index with negative indexing support.
///
/// ## Arguments
///
/// * `rank` - The rank of the tensor.
/// * `idx` - The dimension index to canonicalize.
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
pub fn canonicalize_dim<I>(idx: I, rank: usize, wrap_scalar: bool) -> usize
where
    I: IndexConversion,
{
    canonicalize_index("dimension", idx, rank, wrap_scalar)
}

/// Canonicalizes and bounds checks an index with negative indexing support.
///
/// ## Arguments
///
/// * `name` - The name of the index (for error messages).
/// * `idx` - The index to canonicalize.
/// * `size` - The size of the dimension.
/// * `wrap_scalar` - If true, treat scalar dimensions as having size 1.
///
/// ## Returns
///
/// The canonicalized index.
///
/// ## Panics
///
/// * If `wrap_scalar` is false and the size is 0.
/// * If the index is out of range for the dimension size.
pub fn canonicalize_index<I>(name: &str, idx: I, size: usize, wrap_scalar: bool) -> usize
where
    I: IndexConversion,
{
    let idx = idx.index();

    let rank = if size > 0 {
        size
    } else {
        if !wrap_scalar {
            panic!("{name} has size {size} and index {idx}");
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
        panic!("{name} index {idx} out of range: ({lower}..={upper})");
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
#[must_use]
pub fn wrap_index<I>(idx: I, size: usize) -> usize
where
    I: IndexConversion,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_idx() {
        for idx in 0..3 {
            assert_eq!(wrap_index(idx, 3), idx as usize);
            assert_eq!(wrap_index(idx + 3, 3), idx as usize);
            assert_eq!(wrap_index(idx + 2 * 3, 3), idx as usize);
            assert_eq!(wrap_index(idx - 3, 3), idx as usize);
            assert_eq!(wrap_index(idx - 2 * 3, 3), idx as usize);
        }
    }

    #[test]
    fn test_canonicalize_dim() {
        for idx in 0..3 {
            let wrap_scalar = false;
            assert_eq!(canonicalize_dim(idx, 3, wrap_scalar), idx as usize);
            assert_eq!(
                canonicalize_dim(-(idx + 1), 3, wrap_scalar),
                (3 - (idx + 1)) as usize
            );
        }

        let wrap_scalar = true;
        assert_eq!(canonicalize_dim(0, 0, wrap_scalar), 0);
        assert_eq!(canonicalize_dim(-1, 0, wrap_scalar), 0);
    }

    #[test]
    #[should_panic = "dimension has size 0 and index 0"]
    fn test_canonicalize_error_no_dims() {
        let _d = canonicalize_dim(0, 0, false);
    }

    #[test]
    #[should_panic = "dimension index 3 out of range: (-3..=2)"]
    fn test_canonicalize_error_too_big() {
        let _d = canonicalize_dim(3, 3, false);
    }
    #[test]
    #[should_panic = "dimension index -4 out of range: (-3..=2)"]
    fn test_canonicalize_error_too_small() {
        let _d = canonicalize_dim(-4, 3, false);
    }
}
