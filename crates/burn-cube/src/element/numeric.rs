use crate::{CubeContext, CubeType, PrimitiveVariable};

/// Type that encompasses both (unsigned or signed) integers and floats
/// Used in kernels that should work for both.
pub trait Numeric:
    Clone
    + Copy
    + PrimitiveVariable
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::cmp::PartialOrd
{
    /// Create a new constant numeric.
    ///
    /// Note: since this must work for both integer and float
    /// only the less expressive of both can be created (int)
    /// If a number with decimals is needed, use Float::from_primitive.
    fn lit(val: i64) -> Self;

    /// Expand version of new
    fn lit_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType;
}
