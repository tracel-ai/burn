use crate::frontend::operation::base::binary_expand;
use crate::frontend::{CubeContext, ExpandElement, UInt, BF16, F16, F32, F64, I32, I64};
use crate::ir::Operator;
use crate::{frontend::CubeType, unexpanded};

pub mod add {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, Operator::Add)
    }

    macro_rules! impl_add {
        ($type:ty) => {
            impl core::ops::Add for $type {
                type Output = Self;

                fn add(self, rhs: Self) -> Self::Output {
                    (self.val + rhs.val).into()
                }
            }
        };
    }

    impl_add!(F16);
    impl_add!(BF16);
    impl_add!(F32);
    impl_add!(F64);
    impl_add!(I32);
    impl_add!(I64);
    impl_add!(UInt);
}

pub mod sub {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, Operator::Sub)
    }

    macro_rules! impl_sub {
        ($type:ty) => {
            impl core::ops::Sub for $type {
                type Output = Self;

                fn sub(self, _rhs: Self) -> Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_sub!(F16);
    impl_sub!(BF16);
    impl_sub!(F32);
    impl_sub!(F64);
    impl_sub!(I32);
    impl_sub!(I64);
    impl_sub!(UInt);
}

pub mod mul {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, Operator::Mul)
    }

    macro_rules! impl_mul {
        ($type:ty) => {
            impl core::ops::Mul for $type {
                type Output = Self;

                fn mul(self, _rhs: Self) -> Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_mul!(F16);
    impl_mul!(BF16);
    impl_mul!(F32);
    impl_mul!(F64);
    impl_mul!(I32);
    impl_mul!(I64);
    impl_mul!(UInt);
}

pub mod div {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, Operator::Div)
    }

    macro_rules! impl_div {
        ($type:ty) => {
            impl core::ops::Div for $type {
                type Output = Self;

                fn div(self, _rhs: Self) -> Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_div!(F16);
    impl_div!(BF16);
    impl_div!(F32);
    impl_div!(F64);
    impl_div!(I32);
    impl_div!(I64);
    impl_div!(UInt);
}

pub mod rem {
    use super::*;

    pub fn expand(
        context: &mut CubeContext,
        lhs: ExpandElement,
        rhs: ExpandElement,
    ) -> ExpandElement {
        binary_expand(context, lhs, rhs, Operator::Modulo)
    }

    macro_rules! impl_rem {
        ($type:ty) => {
            impl core::ops::Rem for $type {
                type Output = Self;

                fn rem(self, _rhs: Self) -> Self::Output {
                    unexpanded!()
                }
            }
        };
    }

    impl_rem!(I32);
    impl_rem!(I64);
    impl_rem!(UInt);
}

pub mod and {
    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        binary_expand(context, lhs.into(), rhs.into(), Operator::And)
    }
}

pub mod bitand {
    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        binary_expand(context, lhs.into(), rhs.into(), Operator::BitwiseAnd)
    }

    impl core::ops::BitAnd for UInt {
        type Output = UInt;

        fn bitand(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod or {
    use super::*;

    pub fn expand<L, R>(context: &mut CubeContext, lhs: L, rhs: R) -> ExpandElement
    where
        L: Into<ExpandElement>,
        R: Into<ExpandElement>,
    {
        binary_expand(context, lhs.into(), rhs.into(), Operator::Or)
    }
}

pub mod bitxor {
    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        binary_expand(context, lhs.into(), rhs.into(), Operator::BitwiseXor)
    }

    impl core::ops::BitXor for UInt {
        type Output = UInt;

        fn bitxor(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod shl {
    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        binary_expand(context, lhs.into(), rhs.into(), Operator::ShiftLeft)
    }

    impl core::ops::Shl for UInt {
        type Output = UInt;

        fn shl(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

pub mod shr {
    use super::*;

    pub fn expand<L: Into<ExpandElement>, R: Into<ExpandElement>>(
        context: &mut CubeContext,
        lhs: L,
        rhs: R,
    ) -> ExpandElement {
        binary_expand(context, lhs.into(), rhs.into(), Operator::ShiftRight)
    }

    impl core::ops::Shr for UInt {
        type Output = UInt;

        fn shr(self, _rhs: Self) -> Self::Output {
            unexpanded!()
        }
    }
}

/// For binary functions without special syntax
macro_rules! impl_binary_func {
    ($trait_name:ident, $method_name:ident, $method_name_expand:ident, $operator:expr, $($type:ty),*) => {
        pub trait $trait_name: CubeType + Sized {
            fn $method_name(self, _rhs: Self) -> Self {
                unexpanded!()
            }

            fn $method_name_expand(context: &mut CubeContext, lhs: ExpandElement, rhs: ExpandElement) -> ExpandElement {
                binary_expand(context, lhs, rhs, $operator)
            }
        }

        $(impl $trait_name for $type {})*
    }
}

impl_binary_func!(Powf, powf, powf_expand, Operator::Powf, F16, BF16, F32, F64);
impl_binary_func!(
    Max,
    max,
    max_expand,
    Operator::Max,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
impl_binary_func!(
    Min,
    min,
    min_expand,
    Operator::Min,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
impl_binary_func!(
    Remainder,
    rem,
    rem_expand,
    Operator::Remainder,
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    UInt
);
