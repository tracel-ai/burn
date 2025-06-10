//! A module for dimension indexing utility machinery.

/// A trait for types that can be used as dimension indices.
pub trait ReflectableIndex: Copy + Sized {
    /// Converts the index to an `isize` for internal processing.
    fn as_isize_index(&self) -> isize;
}

impl ReflectableIndex for isize {
    fn as_isize_index(&self) -> isize {
        *self
    }
}

impl ReflectableIndex for usize {
    fn as_isize_index(&self) -> isize {
        *self as isize
    }
}

impl ReflectableIndex for i32 {
    fn as_isize_index(&self) -> isize {
        *self as isize
    }
}

impl ReflectableIndex for u32 {
    fn as_isize_index(&self) -> isize {
        *self as isize
    }
}

/// Canonicalizes and bounds checks a dimension index.
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
    I: ReflectableIndex,
{
    let idx = idx.as_isize_index();

    let rank = if rank > 0 {
        rank
    } else {
        if !wrap_scalar {
            panic!("Dimension specified as {idx} but tensor has no dimensions");
        }
        1
    };

    if idx >= 0 && (idx as usize) < rank {
        return idx as usize;
    }

    let _idx = if idx < 0 { idx + rank as isize } else { idx };

    if _idx < 0 || (_idx as usize) >= rank {
        panic!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})",
            -(rank as isize),
            rank - 1,
            idx
        );
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
pub fn wrap_idx<I>(idx: I, size: usize) -> usize
where
    I: ReflectableIndex,
{
    if size == 0 {
        return 0; // Avoid modulo by zero
    }
    let wrapped = idx.as_isize_index().rem_euclid(size as isize);
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
            assert_eq!(wrap_idx(idx, 3), idx as usize);
            assert_eq!(wrap_idx(idx + 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx + 2 * 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx - 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx - 2 * 3, 3), idx as usize);
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
    #[should_panic = "Dimension specified as 0 but tensor has no dimensions"]
    fn test_canonicalize_error_no_dims() {
        let _d = canonicalize_dim(0, 0, false);
    }

    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got 3)"]
    fn test_canonicalize_error_too_big() {
        let _d = canonicalize_dim(3, 3, false);
    }
    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got -4)"]
    fn test_canonicalize_error_too_small() {
        let _d = canonicalize_dim(-4, 3, false);
    }
}
