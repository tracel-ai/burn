use std::rc::Rc;

use crate::{dialect::Variable, CubeContext, CubeType, ExpandElement, PrimitiveVariable};

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
    fn from_int(val: i64) -> Self {
        <Self as PrimitiveVariable>::from_i64(val)
    }

    /// Expand version of lit
    fn from_int_expand(_context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
        ExpandElement::new(Rc::new(new_var))
    }

}
