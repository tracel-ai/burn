use crate::dialect;
use crate::language::{Array, CubeContext, ExpandElement, UInt};

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
    use crate::language::CubeType;

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

    impl<E: CubeType, I: Into<UInt>> core::ops::IndexMut<I> for Array<E> {
        fn index_mut(&mut self, _index: I) -> &mut Self::Output {
            panic!()
        }
    }
}

pub mod index {
    use crate::language::{operation::base::binary_expand, CubeType};

    use self::dialect::Operator;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, array, index, Operator::Index)
    }

    impl<E: CubeType, I: Into<UInt>> core::ops::Index<I> for Array<E> {
        type Output = E;

        fn index(&self, _index: I) -> &Self::Output {
            panic!()
        }
    }
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
        ($type:ty) => {
            impl core::ops::AddAssign for $type {
                fn add_assign(&mut self, rhs: Self) {
                    self.val += rhs.val
                }
            }
        };
    }

    impl_add_assign!(F16);
    impl_add_assign!(BF16);
    impl_add_assign!(F32);
    impl_add_assign!(F64);
    impl_add_assign!(I32);
    impl_add_assign!(I64);
    impl_add_assign!(UInt);
}
