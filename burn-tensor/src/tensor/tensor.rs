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
    use crate::{tensor::backend::tch::TchTensor, tensor::ops::TensorOpsMul};
    use ::tch::kind::Element as TchElement;

    impl<P: Element + Into<f64> + TchElement, const D: usize> Tensor<P, D> for TchTensor<P, D> {}

    impl<P: Element + Into<f64> + TchElement, const D: usize> Zeros<TchTensor<P, D>>
        for TchTensor<P, D>
    {
        fn zeros(&self) -> TchTensor<P, D> {
            TensorOpsMul::mul_scalar(&self, &P::ZERO)
        }
    }

    impl<P, const D: usize> Ones<TchTensor<P, D>> for TchTensor<P, D>
    where
        P: TchElement + Element + Into<f64>,
    {
        fn ones(&self) -> TchTensor<P, D> {
            let tensor = self.tensor.ones_like();
            let kind = self.kind.clone();
            let shape = self.shape.clone();

            Self {
                tensor,
                kind,
                shape,
            }
        }
    }
}

mod ndarray {
    use super::*;
    use crate::tensor::backend::ndarray::NdArrayTensor;
    use ::ndarray::{Dim, Dimension, LinalgScalar, ScalarOperand};

    impl<P: Element + ScalarOperand + LinalgScalar, const D: usize> Tensor<P, D> for NdArrayTensor<P, D> where
        Dim<[usize; D]>: Dimension
    {
    }

    impl<P: Element, const D: usize> Zeros<Self> for NdArrayTensor<P, D>
    where
        P: Ones<P> + Default + ScalarOperand + LinalgScalar,
        Dim<[usize; D]>: Dimension,
    {
        fn zeros(&self) -> Self {
            TensorOpsMul::mul_scalar(&self, &P::default().zeros())
        }
    }

    impl<P: Element, const D: usize> Ones<Self> for NdArrayTensor<P, D>
    where
        P: Ones<P> + Default + ScalarOperand + LinalgScalar,
        Dim<[usize; D]>: Dimension,
    {
        fn ones(&self) -> NdArrayTensor<P, D> {
            let x = TensorOpsMul::mul_scalar(self, &P::default().zeros());
            let x = TensorOpsAdd::add_scalar(&x, &P::default().ones());
            x
        }
    }
}

mod ad {
    use super::*;
    use crate::tensor::backend::autodiff::ADTensor;

    impl<T: Tensor<P, D>, P: Element, const D: usize> Tensor<P, D> for ADTensor<P, D, T> {}

    impl<T: Tensor<P, D>, P: Element, const D: usize> Zeros<Self> for ADTensor<P, D, T>
    where
        P: Ones<P> + Default,
    {
        fn zeros(&self) -> Self {
            TensorOpsMul::mul_scalar(&self, &P::default().zeros())
        }
    }

    impl<T: Tensor<P, D>, P: Element, const D: usize> Ones<Self> for ADTensor<P, D, T>
    where
        P: Ones<P> + Default,
    {
        fn ones(&self) -> Self {
            let x = TensorOpsMul::mul_scalar(self, &P::default().zeros());
            let x = TensorOpsAdd::add_scalar(&x, &P::default().ones());
            x
        }
    }
}
