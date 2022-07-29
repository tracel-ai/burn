use super::ADTensor;
use crate::tensor::{Backend, Data, Element, Tensor, TensorType};
use rand::distributions::Standard;

#[derive(Debug)]
pub struct ADTensorBackend<E, B> {
    _b: B,
    _e: E,
}

impl<E: Default, B: Backend> Default for ADTensorBackend<E, B> {
    fn default() -> Self {
        Self {
            _b: B::default(),
            _e: E::default(),
        }
    }
}

#[macro_export]
/// Run the macro in score with de name define_impl with backend as argument.
/// Element should be defined in generic as E.
macro_rules! register_ad_backend {
    () => {
        #[cfg(feature = "ndarray")]
        type B1<E> = crate::tensor::backend::ndarray::NdArrayTensorBackend<E>;
        #[cfg(feature = "tch")]
        type B2<E> = crate::tensor::backend::tch::TchTensorCPUBackend<E>;
        type AD<E, B> = crate::tensor::backend::autodiff::ADTensorBackend<E, B>;

        // First order derivative
        #[cfg(feature = "ndarray")]
        define_impl!(B1<E>);
        #[cfg(feature = "tch")]
        define_impl!(B2<E>);

        // Second order derivative
        #[cfg(feature = "ndarray")]
        define_impl!(AD<E, B1<E>>);
        #[cfg(feature = "tch")]
        define_impl!(AD<E, B2<E>>);

        // third order derivative
        #[cfg(feature = "ndarray")]
        define_impl!(AD<E, AD<E, B1<E>>>);
        #[cfg(feature = "tch")]
        define_impl!(AD<E, AD<E, B2<E>>>);
    };
}

macro_rules! define_impl {
    ($b:ty) => {
        impl<E> Backend for ADTensorBackend<E, $b>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            type E = E;

            fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<D, Self>>::T
            where
                Self: TensorType<D, Self>,
            {
                <Self as TensorType<D, Self>>::from_data(data)
            }
        }

        impl<E, const D: usize> TensorType<D, Self> for ADTensorBackend<E, $b>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            type T = ADTensor<E, D, Tensor<D, $b>>;

            fn from_data(data: Data<E, D>) -> Self::T {
                let tensor = <$b as TensorType<D, $b>>::from_data(data);
                let tensor = ADTensor::from_tensor(tensor);
                tensor
            }
        }
    };
}

register_ad_backend!();
