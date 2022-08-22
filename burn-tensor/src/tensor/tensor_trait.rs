use crate::tensor::ops::*;
use half::bf16;
use num_traits::{FromPrimitive, ToPrimitive};
use rand::distributions::uniform::SampleUniform;

pub trait Element:
    Zeros<Self>
    + ElementConversion
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
    fn log_elem(self) -> Self;
}

pub trait ElementConversion {
    fn from_elem<E: ToPrimitive>(elem: E) -> Self;
}

#[cfg(feature = "ndarray")]
pub trait NdArrayElement:
    Element + ndarray::LinalgScalar + ndarray::ScalarOperand + ExpElement + num_traits::FromPrimitive
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
        one $one:expr,
        convert $convert:expr

    ) => {
        impl Element for $float {}

        impl Zeros<$float> for $float {
            fn zeros(&self) -> $float {
                $zero
            }
        }

        impl ElementConversion for $float {
            fn from_elem<E: ToPrimitive>(elem: E) -> Self {
                $convert(&elem)
            }
        }

        impl Ones<$float> for $float {
            fn ones(&self) -> $float {
                $one
            }
        }
    };
    (
        float $float:ident,
        convert $convert:expr
    ) => {
        ad_items!(ty $float, zero 0.0, one 1.0, convert $convert);
    };
    (
        int $int:ident,
        convert $convert:expr
    ) => {
        ad_items!(ty $int, zero 0, one 1, convert $convert);
    };
}

ad_items!(float f64, convert |elem: &dyn ToPrimitive| elem.to_f64().unwrap());
ad_items!(float f32, convert |elem: &dyn ToPrimitive| elem.to_f32().unwrap());
// ad_items!(float bf16, convert |elem: &dyn ToPrimitive| elem.to_f32().unwrap());

ad_items!(int i64, convert |elem: &dyn ToPrimitive| elem.to_i64().unwrap());
ad_items!(int i32, convert |elem: &dyn ToPrimitive| elem.to_i32().unwrap());
ad_items!(int i16, convert |elem: &dyn ToPrimitive| elem.to_i16().unwrap());
ad_items!(int i8, convert |elem: &dyn ToPrimitive| elem.to_i8().unwrap());

ad_items!(int u64, convert |elem: &dyn ToPrimitive| elem.to_u64().unwrap());
ad_items!(int u32, convert |elem: &dyn ToPrimitive| elem.to_u32().unwrap());
ad_items!(int u16, convert |elem: &dyn ToPrimitive| elem.to_u16().unwrap());
ad_items!(int u8, convert |elem: &dyn ToPrimitive| elem.to_u8().unwrap());

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
                fn log_elem(self) -> Self {
                    $elem::ln(self)
                }
            }
        };
        ($elem:ident, $tmp:ident) => {
            impl ExpElement for $elem {
                fn exp_elem(self) -> Self {
                    let tmp = $tmp::exp(self as $tmp);
                    tmp as $elem
                }
                fn log_elem(self) -> Self {
                    let tmp = $tmp::ln(self as $tmp);
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
