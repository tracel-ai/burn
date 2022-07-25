use crate::{
    node::{Ones, Zeros},
    TensorBase, TensorOpsAdd, TensorOpsMatmul, TensorOpsMul, TensorOpsNeg, TensorOpsSub,
    TensorOpsTranspose,
};
use half::f16;

pub trait ADElement:
    Zeros<Self> + Ones<Self> + std::fmt::Debug + Default + 'static + Send + Sync + Copy
{
}

pub trait ADCompatibleTensor<P: ADElement, const D: usize>:
    TensorBase<P, D>
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
        impl ADElement for $float {}
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
    use super::{ADCompatibleTensor, ADElement};
    use crate::{
        backend::tch::TchTensor,
        node::{Ones, Zeros},
        TensorOpsMul,
    };

    impl<P: tch::kind::Element + Into<f64> + ADElement, const D: usize> ADCompatibleTensor<P, D>
        for TchTensor<P, D>
    {
    }

    impl<P: tch::kind::Element + Into<f64> + ADElement, const D: usize> Zeros<TchTensor<P, D>>
        for TchTensor<P, D>
    {
        fn zeros(&self) -> TchTensor<P, D> {
            TensorOpsMul::mul_scalar(&self, &P::ZERO)
        }
    }

    impl<P, const D: usize> Ones<TchTensor<P, D>> for TchTensor<P, D>
    where
        P: ADElement + tch::kind::Element + Into<f64>,
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
