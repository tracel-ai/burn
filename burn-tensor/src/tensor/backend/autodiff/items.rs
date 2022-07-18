use crate::{
    node::{Ones, Zeros},
    FloatTensor,
};

pub trait ADFloat:
    num_traits::Float + Zeros<Self> + Ones<Self> + std::fmt::Debug + Default + 'static
{
}
pub trait ADFloatTensor<P: num_traits::Float, const D: usize>:
    FloatTensor<P, D> + Clone + Zeros<Self> + Ones<Self> + std::fmt::Debug + 'static
{
}

macro_rules! ad_items {
    (
        float $float:ident
    ) => {
        impl ADFloat for $float {}
        impl Zeros<$float> for $float {
            fn zeros(&self) -> $float {
                0.0
            }
        }

        impl Ones<$float> for $float {
            fn ones(&self) -> $float {
                1.0
            }
        }
    };
}

ad_items!(float f64);
ad_items!(float f32);

#[cfg(feature = "tch")]
mod tch {
    use super::{ADFloat, ADFloatTensor};
    use crate::{
        backend::tch::TchTensor,
        node::{Ones, Zeros},
        TensorOpsAdd, TensorOpsMul,
    };
    use num_traits::Float;

    impl<P: tch::kind::Element + Into<f64> + ADFloat, const D: usize> ADFloatTensor<P, D>
        for TchTensor<P, D>
    {
    }

    impl<P: tch::kind::Element + Into<f64>, const D: usize> Zeros<TchTensor<P, D>> for TchTensor<P, D> {
        fn zeros(&self) -> TchTensor<P, D> {
            TensorOpsMul::mul_scalar(&self, &P::ZERO)
        }
    }

    impl<P, const D: usize> Ones<TchTensor<P, D>> for TchTensor<P, D>
    where
        P: tch::kind::Element + Into<f64> + Float + Default,
        P: std::fmt::Debug,
    {
        fn ones(&self) -> TchTensor<P, D> {
            let zeros = self.zeros();
            TensorOpsAdd::add_scalar(&zeros, &P::one())
        }
    }
}
