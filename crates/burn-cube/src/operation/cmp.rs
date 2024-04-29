use crate::operation::base::binary_expand;
use crate::{CubeContext, ExpandElement, Float, Int, UInt};
use burn_jit::gpu::{self};

impl core::cmp::PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.vectorization == other.vectorization
    }
}

impl core::cmp::PartialEq for Int {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.vectorization == other.vectorization
    }
}

impl core::cmp::PartialEq for UInt {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.vectorization == other.vectorization
    }
}

impl core::cmp::Eq for Float {}

impl core::cmp::Eq for Int {}

impl core::cmp::Eq for UInt {}

pub mod ne {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::NotEqual)
    }
}
