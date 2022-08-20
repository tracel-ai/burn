use crate::tensor::ops::*;
use rand::distributions::uniform::SampleUniform;

pub trait Element:
    Zeros<Self>
    + num_traits::cast::FromPrimitive
    + Ones<Self>
    + std::ops::Mul<Self, Output = Self>
    + std::fmt::Debug
    + Default
    + 'static
    + Send
    + Sync
    + Copy
    + SampleUniform
    + std::cmp::PartialOrd<Self>
{
}

#[cfg(feature = "tch")]
pub trait TchElement: Element + tch::kind::Element + Into<f64> {}

pub trait ExpElement {
    fn exp_elem(self) -> Self;
}

#[cfg(feature = "ndarray")]
pub trait NdArrayElement:
    Element + ndarray::LinalgScalar + ndarray::ScalarOperand + ExpElement
{
}

pub trait TensorTrait<P: Element, const D: usize>:
    TensorOpsUtilities<P, D>
    + TensorOpsMatmul<P, D>
    + TensorOpsTranspose<P, D>
    + TensorOpsMul<P, D>
    + TensorOpsDiv<P, D>
    + TensorOpsNeg<P, D>
    + TensorOpsAdd<P, D>
    + TensorOpsSub<P, D>
    + Zeros<Self>
    + Ones<Self>
    + Clone
    + Send
    + Sync
    + std::fmt::Debug
{
}

macro_rules! ad_items {
    (
        ty $float:ident,
        zero $zero:expr,
        one $one:expr
    ) => {
        impl Element for $float {}

        impl Zeros<$float> for $float {
            fn zeros(&self) -> $float {
                $zero
            }
        }

        impl Ones<$float> for $float {
            fn ones(&self) -> $float {
                $one
            }
        }
    };
    (
        float $float:ident
    ) => {
        ad_items!(ty $float, zero 0.0, one 1.0);
    };
    (
        int $int:ident
    ) => {
        ad_items!(ty $int, zero 0, one 1);
    };
}

ad_items!(float f64);
ad_items!(float f32);

ad_items!(int i64);
ad_items!(int i32);
ad_items!(int i16);
ad_items!(int i8);

ad_items!(int u64);
ad_items!(int u32);
ad_items!(int u16);
ad_items!(int u8);

#[cfg(feature = "tch")]
mod tch_elem {
    use super::*;

    impl TchElement for f64 {}
    impl TchElement for f32 {}

    impl TchElement for i32 {}
    impl TchElement for i16 {}

    impl TchElement for u8 {}
}

#[cfg(feature = "ndarray")]
mod ndarray_elem {
    use super::*;

    macro_rules! impl_exp_elem {
        ($elem:ident) => {
            impl ExpElement for $elem {
                fn exp_elem(self) -> Self {
                    $elem::exp(self)
                }
            }
        };
        ($elem:ident, $tmp:ident) => {
            impl ExpElement for $elem {
                fn exp_elem(self) -> Self {
                    let tmp = $tmp::exp(self as $tmp);
                    tmp as $elem
                }
            }
        };
    }

    impl NdArrayElement for f64 {}
    impl_exp_elem!(f64);

    impl NdArrayElement for f32 {}
    impl_exp_elem!(f32);

    impl NdArrayElement for i64 {}
    impl_exp_elem!(i64, f64);

    impl NdArrayElement for i32 {}
    impl_exp_elem!(i32, f32);

    impl NdArrayElement for i16 {}
    impl_exp_elem!(i16, f32);

    impl NdArrayElement for u64 {}
    impl_exp_elem!(u64, f64);

    impl NdArrayElement for u32 {}
    impl_exp_elem!(u32, f32);

    impl NdArrayElement for u16 {}
    impl_exp_elem!(u16, f32);

    impl NdArrayElement for u8 {}
    impl_exp_elem!(u8, f32);
}

mod ad {
    use super::*;
    use crate::tensor::backend::{autodiff::ADTensor, Backend};

    impl<B: Backend, const D: usize> TensorTrait<B::Elem, D> for ADTensor<D, B> {}
}
