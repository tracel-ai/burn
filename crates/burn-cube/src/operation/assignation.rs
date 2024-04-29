use crate::{Array, CubeContext, ExpandElement, UInt};
use burn_jit::gpu::{self};

pub mod assign {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        input: ExpandElement,
        output: ExpandElement,
    ) -> ExpandElement {
        let input = *input;
        let out = *output;

        context.register(gpu::Operator::Assign(gpu::UnaryOperator { input, out }));

        output
    }
}

pub mod index_assign {
    use crate::RuntimeType;

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

    impl<E: RuntimeType, I: Into<UInt>> core::ops::IndexMut<I> for Array<E> {
        fn index_mut(&mut self, index: I) -> &mut Self::Output {
            let index = index.into().val;
            &mut self.vals[index as usize]
        }
    }
}

pub mod index {
    use crate::{operation::base::binary_expand, RuntimeType};

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        array: ExpandElement,
        index: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, array, index, gpu::Operator::Index)
    }

    impl<E: RuntimeType, I: Into<UInt>> core::ops::Index<I> for Array<E> {
        type Output = E;

        fn index(&self, index: I) -> &Self::Output {
            let index = index.into().val;
            &self.vals[index as usize]
        }
    }
}
