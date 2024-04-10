use crate::{CubeContext, ExpandElement, Float, Int, UInt};
use burn_jit::gpu::{self, Variable};

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

fn binary_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(gpu::BinaryOperator) -> gpu::Operator,
{
    let lhs: Variable = *lhs;
    let rhs: Variable = *rhs;

    let item = lhs.item();
    let out = context.create_local(item);
    let out_var = *out;

    let op = func(gpu::BinaryOperator {
        lhs,
        rhs,
        out: out_var,
    });

    context.scope.register(op);

    out
}
