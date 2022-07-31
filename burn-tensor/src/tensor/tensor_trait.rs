use crate::tensor::ops::*;
use rand::distributions::uniform::SampleUniform;

pub trait BasicElement:
    Zeros<Self> + Ones<Self> + std::fmt::Debug + Default + 'static + Send + Sync + Copy + SampleUniform
{
}
#[cfg(all(feature = "tch", feature = "ndarray"))]
pub trait Element:
    Sized
    + BasicElement
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + tch::kind::Element
    + Into<f64>
{
}
#[cfg(all(feature = "tch", not(feature = "ndarray")))]
pub trait Element: BasicElement + tch::kind::Element + Into<f64> {}

#[cfg(all(feature = "ndarray", not(feature = "tch")))]
pub trait Element: BasicElement + ndarray::LinalgScalar + ndarray::ScalarOperand {}

pub trait TensorTrait<P: Element, const D: usize>:
    TensorOpsUtilities<P, D>
    + TensorOpsMatmul<P, D>
    + TensorOpsTranspose<P, D>
    + TensorOpsMul<P, D>
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
        impl BasicElement for $float {}
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

#[cfg(not(feature = "tch"))]
ad_items!(int i64);
ad_items!(int i32);
ad_items!(int i16);
ad_items!(int i8);

#[cfg(not(feature = "tch"))]
ad_items!(int u64);
#[cfg(not(feature = "tch"))]
ad_items!(int u32);
#[cfg(not(feature = "tch"))]
ad_items!(int u16);
ad_items!(int u8);

mod ad {
    use super::*;
    use crate::tensor::backend::{autodiff::ADTensor, Backend};

    impl<B: Backend, const D: usize> TensorTrait<B::Elem, D> for ADTensor<D, B> {}
}
