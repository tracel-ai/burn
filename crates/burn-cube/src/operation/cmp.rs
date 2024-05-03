use crate::operation::base::cmp_expand;
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

impl core::cmp::PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.val.partial_cmp(&other.val) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.vectorization.partial_cmp(&other.vectorization)
    }
}

impl core::cmp::PartialOrd for Int {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.val.partial_cmp(&other.val) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.vectorization.partial_cmp(&other.vectorization)
    }
}

impl core::cmp::PartialOrd for UInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.val.partial_cmp(&other.val) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.vectorization.partial_cmp(&other.vectorization)
    }
}

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
