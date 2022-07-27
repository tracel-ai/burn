use crate::tensor::ops::*;
use half::bf16;
use half::f16;

pub trait Element:
    Zeros<Self> + Ones<Self> + std::fmt::Debug + Default + 'static + Send + Sync + Copy
{
}

pub trait Tensor<P: Element, const D: usize>:
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
    + 'static
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

ad_items!(ty f16, zero f16::from_f32(0.0), one f16::from_f32(1.0));
ad_items!(ty bf16, zero bf16::from_f32(0.0), one bf16::from_f32(1.0));

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
mod tch {
    use super::*;
    use crate::tensor::backend::tch::TchTensor;
    use ::tch::kind::Element as TchElement;

    impl<P: Element + Into<f64> + TchElement, const D: usize> Tensor<P, D> for TchTensor<P, D> {}
}

mod ndarray {
    use super::*;
    use crate::tensor::backend::ndarray::NdArrayTensor;
    use ::ndarray::{Dim, Dimension, LinalgScalar, ScalarOperand};

    impl<P: Element + ScalarOperand + LinalgScalar, const D: usize> Tensor<P, D> for NdArrayTensor<P, D> where
        Dim<[usize; D]>: Dimension
    {
    }
}

mod ad {
    use super::*;
    use crate::tensor::backend::autodiff::ADTensor;

    impl<T: Tensor<P, D>, P: Element, const D: usize> Tensor<P, D> for ADTensor<P, D, T> {}
}
