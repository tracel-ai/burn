use crate::operation::base::cmp_expand;
use crate::{CubeContext, ExpandElement, UInt, BF16, F16, F32, F64, I32, I64};
use burn_jit::gpu::{self};

macro_rules! impl_cmp {
    ($type:ty) => {
        impl core::cmp::PartialEq for $type {
            fn eq(&self, other: &Self) -> bool {
                self.val == other.val && self.vectorization == other.vectorization
            }
        }
        impl core::cmp::Eq for $type {}
        impl core::cmp::PartialOrd for $type {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                match self.val.partial_cmp(&other.val) {
                    Some(core::cmp::Ordering::Equal) => {}
                    ord => return ord,
                }
                self.vectorization.partial_cmp(&other.vectorization)
            }
        }
    };
}

impl_cmp!(F16);
impl_cmp!(BF16);
impl_cmp!(F32);
impl_cmp!(F64);
impl_cmp!(I32);
impl_cmp!(I64);
impl_cmp!(UInt);

pub mod ne {

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        cmp_expand(context, lhs, rhs, gpu::Operator::NotEqual)
    }
}

pub mod gt {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        cmp_expand(context, lhs, rhs, gpu::Operator::Greater)
    }
}

pub mod lt {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        cmp_expand(context, lhs, rhs, gpu::Operator::Lower)
    }
}

pub mod add_assign {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        cmp_expand(context, lhs, rhs, gpu::Operator::Add)
    }
}
