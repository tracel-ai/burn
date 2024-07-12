use crate::{
    frontend::{CubeContext, CubeType, ExpandElement, UInt, BF16, F16, F32, F64, I32, I64},
    ir::Operator,
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

impl_unary_func!(
    Abs,
    abs,
    __expand_abs,
    Operator::Abs,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
impl_unary_func!(Exp, exp, __expand_exp, Operator::Exp, F16, BF16, F32, F64);
impl_unary_func!(Log, log, __expand_log, Operator::Log, F16, BF16, F32, F64);
impl_unary_func!(
    Log1p,
    log1p,
    __expand_log1p,
    Operator::Log1p,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(Cos, cos, __expand_cos, Operator::Cos, F16, BF16, F32, F64);
impl_unary_func!(Sin, sin, __expand_sin, Operator::Sin, F16, BF16, F32, F64);
impl_unary_func!(
    Tanh,
    tanh,
    __expand_tanh,
    Operator::Tanh,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(
    Sqrt,
    sqrt,
    __expand_sqrt,
    Operator::Sqrt,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(
    Floor,
    floor,
    __expand_floor,
    Operator::Floor,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(
    Ceil,
    ceil,
    __expand_ceil,
    Operator::Ceil,
    F16,
    BF16,
    F32,
    F64
);
impl_unary_func!(Erf, erf, __expand_erf, Operator::Erf, F16, BF16, F32, F64);
impl_unary_func!(
    Recip,
    recip,
    __expand_recip,
    Operator::Recip,
    F16,
    BF16,
    F32,
    F64
);
