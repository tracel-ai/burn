use super::ADTensor;
use crate::tensor::ops::*;
use crate::tensor::{Backend, Data, Element, Shape, Tensor, TensorType};
use rand::distributions::Standard;

// #[derive(Debug)]
// pub struct ADBackend<E, B> {
//     _b: B,
//     _e: E,
// }
//
// impl<E: Default, B: Backend> Default for ADBackend<E, B> {
//     fn default() -> Self {
//         Self {
//             _b: B::default(),
//             _e: E::default(),
//         }
//     }
// }

#[derive(Debug)]
pub struct ADBackend<E, B> {
    _b: B,
    _e: E,
}

impl<E: Default, B: Backend> Default for ADBackend<E, B> {
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
        type B1<E> = crate::tensor::backend::ndarray::NdArrayBackend<E>;
        #[cfg(feature = "tch")]
        type B2<E> = crate::tensor::backend::tch::TchBackend<E>;
        type AD<E, B> = crate::tensor::backend::autodiff::ADBackend<E, B>;

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
        impl<E> Backend for ADBackend<E, $b>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            type E = E;
            type Device = <$b as Backend>::Device;

            fn name() -> String {
                format!("AD Backend using {}", <$b as Backend>::name())
            }
        }

        impl<E, const D: usize> TensorType<D, Self> for ADBackend<E, $b>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            type T = ADTensor<E, D, Tensor<D, $b>>;

            fn from_data(data: Data<E, D>, device: <$b as Backend>::Device) -> Self::T {
                let tensor = <$b as TensorType<D, $b>>::from_data(data, device);
                let tensor = ADTensor::from_tensor(tensor);
                tensor
            }
        }
        impl<E> ADBackend<E, $b>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            pub fn from_tensor<const D: usize>(tensor: Tensor<D, $b>) -> Tensor<D, Self> {
                let tensor = ADTensor::from_tensor(tensor);
                tensor
            }
        }
    };
}
impl<E, B> Backend for ADBackend<E, B>
where
    E: Element,
    B: Backend<E = E> + 'static,
    Standard: rand::distributions::Distribution<E>,
{
    type E = E;
    type Device = <B as Backend>::Device;

    fn name() -> String {
        format!("AD Backend using {}", <B as Backend>::name())
    }
}

impl<E, const D: usize, B> TensorType<D, Self> for ADBackend<E, B>
where
    E: Element,
    B: Backend<E = E> + TensorType<D, B> + 'static,
    Standard: rand::distributions::Distribution<E>,
{
    type T = ADTensor<E, D, Tensor<D, B>>;

    fn from_data(data: Data<E, D>, device: <B as Backend>::Device) -> Self::T {
        let tensor = <B as TensorType<D, B>>::from_data(data, device);
        let tensor = ADTensor::from_tensor(tensor);
        tensor
    }
}

// register_ad_backend!();
