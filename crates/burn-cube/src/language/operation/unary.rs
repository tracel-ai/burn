use crate::{
    dialect::Operator, unexpanded, CubeContext, CubeType, ExpandElement, UInt, BF16, F16, F32, F64,
    I32, I64,
};

use super::base::unary_expand;

pub trait Abs: CubeType + Sized {
    fn abs(self) -> Self {
        unexpanded!()
    }

    fn abs_expand(context: &mut CubeContext, x: ExpandElement) -> ExpandElement {
        unary_expand(context, x, Operator::Abs)
    }
}

impl Abs for F16 {}
impl Abs for BF16 {}
impl Abs for F32 {}
impl Abs for F64 {}
impl Abs for I32 {}
impl Abs for I64 {}
impl Abs for UInt {}
