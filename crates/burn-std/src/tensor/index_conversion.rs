//! # Common Index Coercions
//!
//! This module contains common index coercions that can be used to implement
//! various indexing operations.

use super::indexing::IndexWrap;
use crate::Shape;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Types which can be converted to a `usize` Size.
pub trait AsSize: Debug + Copy + Sized {
    /// Convert to a `usize` Size.
    fn as_size(self) -> usize;
}

impl<T> AsSize for &T
where
    T: AsSize,
{
    fn as_size(self) -> usize {
        (*self).as_size()
    }
}

macro_rules! gen_as_size {
    ($ty:ty) => {
        impl AsSize for $ty {
            fn as_size(self) -> usize {
                self.try_into()
                    .unwrap_or_else(|_| panic!(
                        "Unable to convert value to usize: {}_{}",
                        self,
                        stringify!($ty)))
            }
        }
    };
    ($($ty:ty),*) => {$(gen_as_size!($ty);)*};
}

gen_as_size!(usize, isize, i64, u64, i32, u32, i16, u16, i8, u8);

/// Helper trait for implementing indexing with support for negative indices.
///
/// # Example
/// ```rust
/// use burn_std::AsIndex;
///
/// fn example<I: AsIndex, const D: usize>(dim: I, size: usize) -> isize {
///    let dim: usize = dim.expect_dim_index(D);
///    unimplemented!()
/// }
/// ```
pub trait AsIndex: Debug + Copy + Sized {
    /// Converts into an `isize` index.
    fn as_index(self) -> isize;

    /// Short-form [`IndexWrap::expect_index(idx, size)`].
    fn expect_elem_index(self, size: usize) -> usize {
        IndexWrap::expect_elem(self, size)
    }

    /// Short-form [`IndexWrap::expect_dim(idx, size)`].
    fn expect_dim_index(self, size: usize) -> usize {
        IndexWrap::expect_dim(self, size)
    }
}

impl<T> AsIndex for &T
where
    T: AsIndex,
{
    fn as_index(self) -> isize {
        (*self).as_index()
    }
}

macro_rules! gen_as_index {
    ($ty:ty) => {
        impl AsIndex for $ty {
            fn as_index(self) -> isize {
                self as isize
            }
        }
    };
    ($($ty:ty),*) => {$(gen_as_index!($ty);)*};
}

gen_as_index!(usize, isize, i64, u64, i32, u32, i16, u16, i8, u8);

/// Marker trait for sources of reshape arguments.
pub trait RankedReshapeArgsSource<const R: usize>: IntoIterator where Self::Item: AsIndex {}

impl<const R: usize> RankedReshapeArgsSource<R> for Shape {}

impl<I: AsIndex, const R: usize> RankedReshapeArgsSource<R> for [I; R] {}

impl<I: AsIndex, const R: usize> RankedReshapeArgsSource<R> for &[I; R] {}

impl<I: AsIndex, const R: usize> RankedReshapeArgsSource<R> for &[I] {}

impl<I: AsIndex, const R: usize> RankedReshapeArgsSource<R> for Vec<I> {}

impl<I: AsIndex, const R: usize> RankedReshapeArgsSource<R> for &Vec<I> {}

/// Trait for building ranked reshape methods.
///
/// # Example
/// ```rust,ignore
/// impl<const R: usize> Tensor<R> {
///   pub fn reshape<const R2: usize, S: RankedReshapeArgs<R2>>(self, shape: S) -> Tensor<R2> {
///       // Convert reshape args to shape
///       let shape = shape.eval_shape::<R>(self.shape());
///       Tensor::new(K::reshape(self.primitive, shape))
///   }
/// }
/// ```
pub trait RankedReshapeArgs<const TARGET_RANK: usize> {
    /// Evaluates the reshape args, against the source [`Shape`].
    fn eval_shape<const SOURCE_RANK: usize>(self, source: Shape) -> Shape;
}

impl<const TARGET_RANK: usize, T> RankedReshapeArgs<TARGET_RANK> for T
where
    T: RankedReshapeArgsSource<TARGET_RANK>,
    T::Item: AsIndex,
{
    fn eval_shape<const SOURCE_RANK: usize>(self, source: Shape) -> Shape {
        source.reshape(self).expect("invalid reshape")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_size() {
        assert_eq!(1_usize.as_size(), 1_usize);
        assert_eq!(1_isize.as_size(), 1_usize);
        assert_eq!(1_i64.as_size(), 1_usize);
        assert_eq!(1_u64.as_size(), 1_usize);
        assert_eq!(1_i32.as_size(), 1_usize);
        assert_eq!(1_u32.as_size(), 1_usize);
        assert_eq!(1_i16.as_size(), 1_usize);
        assert_eq!(1_u16.as_size(), 1_usize);
        assert_eq!(1_i8.as_size(), 1_usize);
        assert_eq!(1_u8.as_size(), 1_usize);

        assert_eq!((&1_usize).as_size(), 1_usize);
    }

    #[test]
    #[should_panic(expected = "Unable to convert value to usize: -1_isize")]
    fn test_as_size_isize_panic() {
        (-1_isize).as_size();
    }
    #[test]
    #[should_panic(expected = "Unable to convert value to usize: -1_i64")]
    fn test_as_size_i64() {
        (-1_i64).as_size();
    }

    #[test]
    #[should_panic(expected = "Unable to convert value to usize: -1_i32")]
    fn test_as_size_i32() {
        (-1_i32).as_size();
    }

    #[test]
    #[should_panic(expected = "Unable to convert value to usize: -1_i16")]
    fn test_as_size_i16() {
        (-1_i16).as_size();
    }

    #[test]
    #[should_panic(expected = "Unable to convert value to usize: -1_i8")]
    fn test_as_size_i8() {
        (-1_i8).as_size();
    }

    #[test]
    fn test_as_index() {
        assert_eq!(1_usize.as_index(), 1_isize);
        assert_eq!(1_isize.as_index(), 1_isize);
        assert_eq!(1_i64.as_index(), 1_isize);
        assert_eq!(1_u64.as_index(), 1_isize);
        assert_eq!(1_i32.as_index(), 1_isize);
        assert_eq!(1_u32.as_index(), 1_isize);
        assert_eq!(1_i16.as_index(), 1_isize);
        assert_eq!(1_u16.as_index(), 1_isize);
        assert_eq!(1_i8.as_index(), 1_isize);
        assert_eq!(1_u8.as_index(), 1_isize);

        assert_eq!((&1_usize).as_index(), 1_isize);

        assert_eq!(-1_isize.as_index(), -1_isize);
        assert_eq!(-1_i64.as_index(), -1_isize);
        assert_eq!(-1_i32.as_index(), -1_isize);
        assert_eq!(-1_i16.as_index(), -1_isize);
        assert_eq!(-1_i8.as_index(), -1_isize);
    }
}
