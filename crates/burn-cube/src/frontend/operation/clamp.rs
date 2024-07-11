use crate::{
    ir::{ClampOperator, Operator},
    prelude::{CubeContext, CubePrimitive, UInt, BF16, F16, F32, F64, I32, I64},
    unexpanded,
};

use super::unary_expand;

pub trait Clamp: CubePrimitive + Sized {
    /// Clamp the input value between the max and min values provided.
    #[allow(unused_variables)]
    fn clamp(input: Self, min_value: Self, max_value: Self) -> Self {
        unexpanded!()
    }
    fn __expand_clamp(
        context: &mut CubeContext,
        input: Self::ExpandType,
        min_value: Self::ExpandType,
        max_value: Self::ExpandType,
    ) -> Self::ExpandType {
        unary_expand(context, input, |op| {
            Operator::Clamp(ClampOperator {
                input: op.input,
                min_value: *min_value,
                max_value: *max_value,
                out: op.out,
            })
        })
    }
}

impl Clamp for F16 {}
impl Clamp for BF16 {}
impl Clamp for F32 {}
impl Clamp for F64 {}
impl Clamp for I32 {}
impl Clamp for I64 {}
impl Clamp for UInt {}
