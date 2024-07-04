use crate::{
    frontend::{CubeContext, CubeType, ExpandElement, UInt, BF16, F16, F32, F64, I32, I64},
    ir::{ClampOperator, Operator},
    prelude::CubePrimitive,
    unexpanded,
};

use super::base::unary_expand;

pub mod not {
    use super::*;

    pub fn expand(context: &mut CubeContext, x: ExpandElement) -> ExpandElement {
        unary_expand(context, x, Operator::Not)
    }
}

macro_rules! impl_unary_func {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubeType + Sized {
            fn $method_name(_input: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(context: &mut CubeContext, x: ExpandElement) -> ExpandElement {
                unary_expand(context, x, $operator)
            }
        }

        $(impl $trait_name for $type {})*
    }
}

pub trait Clamp: CubePrimitive + Sized {
    /// Clamp the input value between the max and min values provided.
    #[allow(unused_variables)]
    fn clamp(input: Self, max_value: Self, min_value: Self) -> Self {
        unexpanded!()
    }
    fn clamp_expand(
        context: &mut CubeContext,
        input: Self::ExpandType,
        max_value: Self::ExpandType,
        min_value: Self::ExpandType,
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

impl_unary_func!(
    Abs,
    abs,
    abs_expand,
    Operator::Abs,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
impl_unary_func!(Exp, exp, exp_expand, Operator::Exp, F16, BF16, F32, F64);
impl_unary_func!(Log, log, log_expand, Operator::Log, F16, BF16, F32, F64);
impl_unary_func!(
    Log1p,
    log1p,
    log1p_expand,
    Operator::Log1p,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(Cos, cos, cos_expand, Operator::Cos, F16, BF16, F32, F64);
impl_unary_func!(Sin, sin, sin_expand, Operator::Sin, F16, BF16, F32, F64);
impl_unary_func!(Tanh, tanh, tanh_expand, Operator::Tanh, F16, BF16, F32, F64);
impl_unary_func!(Sqrt, sqrt, sqrt_expand, Operator::Sqrt, F16, BF16, F32, F64);
impl_unary_func!(
    Floor,
    floor,
    floor_expand,
    Operator::Floor,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(Ceil, ceil, ceil_expand, Operator::Ceil, F16, BF16, F32, F64);
impl_unary_func!(Erf, erf, erf_expand, Operator::Erf, F16, BF16, F32, F64);
impl_unary_func!(
    Recip,
    recip,
    recip_expand,
    Operator::Recip,
    F16,
    BF16,
    F32,
    F64
);
