use crate::language::{Array, CubeContext, ExpandElement, Tensor, UInt};
use crate::{dialect, unexpanded};

pub mod assign {
    use self::dialect::{Operator, UnaryOperator};

    use super::*;

    pub fn expand(context: &mut CubeContext, input: ExpandElement, output: ExpandElement) {
        let input = *input;
        let out = *output;

        context.register(Operator::Assign(UnaryOperator { input, out }));
    }
}

pub mod index_assign {
    use crate::{language::CubeType, unexpanded};

    use self::dialect::{BinaryOperator, Operator};

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
        value: ExpandElement,
    ) -> ExpandElement {
        context.register(Operator::IndexAssign(BinaryOperator {
            lhs: *index,
            rhs: *value,
            out: *array,
        }));
        array
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubeType, I: Into<UInt>> core::ops::IndexMut<I> for $type<E> {
                fn index_mut(&mut self, _index: I) -> &mut Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
}

pub mod index {
    use crate::{
        language::{operation::base::binary_expand, CubeType},
        unexpanded,
    };

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, array, index, Operator::Index)
    }

    macro_rules! impl_index {
        ($type:ident) => {
            impl<E: CubeType, I: Into<UInt>> core::ops::Index<I> for $type<E> {
                type Output = E;

                fn index(&self, _index: I) -> &Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_index!(Array);
    impl_index!(Tensor);
}

pub mod add_assign_op {
    use crate::language::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        assign_op_expand(context, lhs, rhs, Operator::Add)
    }

    macro_rules! impl_add_assign {
        ($($type:ty),*) => {
            $(impl core::ops::AddAssign for $type {
                fn add_assign(&mut self, _rhs: Self) {
                    unexpanded!()
                }
            })*
        };
    }

    impl_add_assign!(F16, BF16, F32, F64, I32, I64, UInt);
}

pub mod sub_assign_op {
    use crate::language::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        assign_op_expand(context, lhs, rhs, Operator::Sub)
    }

    macro_rules! impl_add_assign {
        ($($type:ty),*) => {
            $(impl core::ops::SubAssign for $type {
                fn sub_assign(&mut self, _rhs: Self) {
                    unexpanded!()
                }
            })*
        };
    }

    impl_add_assign!(F16, BF16, F32, F64, I32, I64, UInt);
}

pub mod mul_assign_op {
    use crate::language::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        assign_op_expand(context, lhs, rhs, Operator::Mul)
    }

    macro_rules! impl_add_assign {
        ($($type:ty),*) => {
            $(impl core::ops::MulAssign for $type {
                fn mul_assign(&mut self, _rhs: Self) {
                    unexpanded!()
                }
            })*
        };
    }

    impl_add_assign!(F16, BF16, F32, F64, I32, I64, UInt);
}

pub mod div_assign_op {
    use crate::language::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        assign_op_expand(context, lhs, rhs, Operator::Div)
    }

    macro_rules! impl_add_assign {
        ($($type:ty),*) => {
            $(impl core::ops::DivAssign for $type {
                fn div_assign(&mut self, _rhs: Self) {
                    unexpanded!()
                }
            })*
        };
    }

    impl_add_assign!(F16, BF16, F32, F64, I32, I64, UInt);
}
