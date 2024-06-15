use crate::frontend::{Array, CubeContext, ExpandElement, SharedMemory, Tensor, UInt};
use crate::{ir, unexpanded};

pub mod assign {
    use self::ir::{Operator, UnaryOperator};

    use super::*;

    pub fn expand<I: Into<ExpandElement>, O: Into<ExpandElement>>(
        context: &mut CubeContext,
        input: I,
        output: O,
    ) {
        let input = *input.into();
        let out = *output.into();

        context.register(Operator::Assign(UnaryOperator { input, out }));
    }
}

pub mod index_assign {
    use crate::{frontend::CubeType, unexpanded};

    use self::ir::{BinaryOperator, Operator, Variable};

    use super::*;

    pub fn expand<A: Into<ExpandElement>, I: Into<ExpandElement>, V: Into<ExpandElement>>(
        context: &mut CubeContext,
        array: A,
        index: I,
        value: V,
    ) -> ExpandElement {
        let array = array.into();
        let index: Variable = *index.into();
        let index = match index {
            Variable::ConstantScalar(val, _) => Variable::ConstantScalar(val, ir::Elem::UInt),
            _ => index,
        };
        context.register(Operator::IndexAssign(BinaryOperator {
            lhs: index,
            rhs: *value.into(),
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
    impl_index!(SharedMemory);
}

pub mod index {
    use crate::{
        frontend::{
            operation::base::{binary_expand, binary_expand_no_vec},
            CubeType,
        },
        unexpanded,
    };

    use self::ir::{Operator, Variable};

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        array: L,
        index: R,
    ) -> ExpandElement {
        let index = index.into();
        let index_var: Variable = *index;
        let index = match index_var {
            Variable::ConstantScalar(val, _) => {
                ExpandElement::Plain(Variable::ConstantScalar(val, ir::Elem::UInt))
            }
            _ => index,
        };
        let array = array.into();
        let var: Variable = *array;
        match var {
            Variable::Local(_, _, _) => {
                binary_expand_no_vec(context, array, index, Operator::Index)
            }
            _ => binary_expand(context, array, index, Operator::Index),
        }
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
    impl_index!(SharedMemory);
}

pub mod add_assign_op {
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::ir::Operator;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Add)
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
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::ir::Operator;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Sub)
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
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::ir::Operator;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Mul)
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
    use crate::frontend::{operation::base::assign_op_expand, BF16, F16, F32, F64, I32, I64};

    use self::ir::Operator;

    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        assign_op_expand(context, lhs.into(), rhs.into(), Operator::Div)
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
