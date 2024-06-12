use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement};
use crate::ir::{Item, Variable};
use crate::{
    frontend::{index_assign, Abs, Max, Min, Remainder},
    unexpanded,
};

use super::{LaunchArg, Vectorized};

/// Type that encompasses both (unsigned or signed) integers and floats
/// Used in kernels that should work for both.
pub trait Numeric:
    Vectorized
    + Copy
    + LaunchArg
    + CubeElem
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::cmp::PartialOrd
    + Abs
    + Max
    + Min
    + Remainder
{
    /// Create a new constant numeric.
    ///
    /// Note: since this must work for both integer and float
    /// only the less expressive of both can be created (int)
    /// If a number with decimals is needed, use Float::new.
    ///
    /// This method panics when unexpanded. For creating an element
    /// with a val, use the new method of the sub type.
    fn from_int(_val: i64) -> Self {
        unexpanded!()
    }

    /// Expand version of from_int
    fn from_int_expand(_context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
        ExpandElement::Plain(new_var)
    }

    fn from_vec(_vec: &[i64]) -> Self {
        unexpanded!()
    }

    fn from_vec_expand(context: &mut CubeContext, vec: &[i64]) -> <Self as CubeType>::ExpandType {
        let mut new_var = context.create_local(Item::vectorized(Self::as_elem(), vec.len() as u8));
        for (i, element) in vec.iter().enumerate() {
            new_var = index_assign::expand(context, new_var, i, *element);
        }

        new_var
    }
}
