use crate::dialect::{Item, Variable};
use crate::index_assign;
use crate::language::{CubeContext, CubeType, ExpandElement, PrimitiveVariable};
use std::rc::Rc;

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

    /// Expand version of from_int
    fn from_int_expand(_context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
        ExpandElement::new(Rc::new(new_var))
    }

    fn from_vec(vec: &[i64]) -> Self {
        <Self as PrimitiveVariable>::from_i64_vec(&vec)
    }

    fn from_vec_expand(context: &mut CubeContext, vec: &[i64]) -> <Self as CubeType>::ExpandType {
        let mut new_var = context.create_local(Item {
            elem: Self::into_elem(),
            vectorization: (vec.len() as u8).into(),
        });
        for (i, element) in vec.iter().enumerate() {
            new_var = index_assign::expand(context, new_var, i.into(), (*element).into());
        }

        new_var
    }
}
