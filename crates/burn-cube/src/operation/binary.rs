use crate::operation::base::binary_expand;
use crate::{CubeContext, ExpandElement, Float, Int, UInt};
use burn_jit::gpu::{self};

pub mod add {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Add)
    }

    impl core::ops::Add for Float {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Float::new(self.val + rhs.val, 1)
        }
    }

    impl core::ops::Add for Int {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Int::new(self.val + rhs.val, 1)
        }
    }

    impl core::ops::Add for UInt {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            UInt::new(self.val + rhs.val, 1)
        }
    }
}

pub mod sub {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Sub)
    }

    impl core::ops::Sub for Float {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Float::new(self.val - rhs.val, 1)
        }
    }

    impl core::ops::Sub for Int {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Int::new(self.val - rhs.val, 1)
        }
    }

    impl core::ops::Sub for UInt {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            UInt::new(self.val - rhs.val, 1)
        }
    }
}

pub mod mul {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Mul)
    }

    impl core::ops::Mul for Float {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Float::new(self.val * rhs.val, 1)
        }
    }

    impl core::ops::Mul for Int {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Int::new(self.val * rhs.val, 1)
        }
    }

    impl core::ops::Mul for UInt {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            UInt::new(self.val * rhs.val, 1)
        }
    }
}

pub mod div {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Div)
    }

    impl core::ops::Div for Float {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Float::new(self.val / rhs.val, 1)
        }
    }

    impl core::ops::Div for Int {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Int::new(self.val / rhs.val, 1)
        }
    }

    impl core::ops::Div for UInt {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            UInt::new(self.val / rhs.val, 1)
        }
    }
}

pub mod rem {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Modulo)
    }

    impl core::ops::Rem for Float {
        type Output = Self;

        fn rem(self, rhs: Self) -> Self::Output {
            Float::new(self.val % rhs.val, 1)
        }
    }

    impl core::ops::Rem for Int {
        type Output = Self;

        fn rem(self, rhs: Self) -> Self::Output {
            Int::new(self.val % rhs.val, 1)
        }
    }

    impl core::ops::Rem for UInt {
        type Output = Self;

        fn rem(self, rhs: Self) -> Self::Output {
            UInt::new(self.val % rhs.val, 1)
        }
    }
}

pub mod and {
    use crate::Bool;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::And)
    }

    impl core::ops::BitAnd for Bool {
        type Output = Bool;

        fn bitand(self, rhs: Self) -> Self::Output {
            Bool::new(self.val && rhs.val, 1)
        }
    }
}

pub mod or {
    use crate::Bool;

    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, gpu::Operator::Or)
    }

    impl core::ops::BitOr for Bool {
        type Output = Bool;

        fn bitor(self, rhs: Self) -> Self::Output {
            Bool::new(self.val || rhs.val, 1)
        }
    }
}
