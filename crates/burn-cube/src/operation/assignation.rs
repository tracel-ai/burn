use crate::{Array, CubeContext, ExpandElement, UInt};
use burn_jit::gpu::{self};

pub mod assign {
    use super::*;

    pub fn expand(context: &mut CubeContext, input: ExpandElement, output: ExpandElement) {
        let input = *input;
        let out = *output;

        context.register(gpu::Operator::Assign(gpu::UnaryOperator { input, out }));
    }
}

pub mod index_assign {
    use crate::CubeType;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
        value: ExpandElement,
    ) {
        context.register(gpu::Operator::IndexAssign(gpu::BinaryOperator {
            lhs: *index,
            rhs: *value,
            out: *array,
        }))
    }

    impl<E: CubeType, I: Into<UInt>> core::ops::IndexMut<I> for Array<E> {
        fn index_mut(&mut self, index: I) -> &mut Self::Output {
            let index = index.into().val;
            &mut self.vals[index as usize]
        }
    }
}

pub mod index {
    use crate::{operation::base::binary_expand, CubeType};

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, array, index, gpu::Operator::Index)
    }

    impl<E: CubeType, I: Into<UInt>> core::ops::Index<I> for Array<E> {
        type Output = E;

        fn index(&self, index: I) -> &Self::Output {
            let index = index.into().val;
            &self.vals[index as usize]
        }
    }
}

pub mod add_assign_op {
    use crate::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        assign_op_expand(context, lhs, rhs, gpu::Operator::Add)
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
