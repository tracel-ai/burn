// use super::ADTensor;
// use crate::tensor::backend::Backend;
// use rand::distributions::Standard;
//
// #[derive(Clone, Debug, Default)]
// pub struct ADBackend<B: Backend> {
//     _b: B,
// }
//
// impl<B: Backend> Backend for ADBackend<B>
// where
//     Standard: rand::distributions::Distribution<B::Elem>,
// {
//     type Device = B::Device;
//     type Elem = B::Elem;
//     type Tensor<const D: usize> = ADTensor<D, B>;
// }

use super::ADTensor;
use crate::graph::grad::Gradients;
use crate::tensor::Data;
use crate::tensor::{
    backend::{ADBackend, Backend},
    Element,
};
use rand::distributions::Standard;

macro_rules! define_impl {
    (
        name: $name:ident,
        backend: $backend:ty
    ) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name<E> {
            _b: $backend,
        }

        impl<E: Element> Backend for $name<E>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            type Device = <$backend as Backend>::Device;
            type Elem = E;
            type Tensor<const D: usize> = ADTensor<D, $backend>;

            fn from_data<const D: usize>(
                data: Data<Self::Elem, D>,
                device: Self::Device,
            ) -> Self::Tensor<D> {
                let tensor = <$backend as Backend>::from_data(data, device);
                ADTensor::from_tensor(tensor)
            }

            fn ad_enabled() -> bool {
                true
            }

            fn name() -> String {
                format!("autodiff<{}>", <$backend as Backend>::name())
            }
        }

        impl<E: Element> ADBackend for $name<E>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            type InnerBackend = $backend;

            fn backward<const D: usize>(tensor: &Self::Tensor<D>) -> Gradients {
                tensor.backward()
            }
            fn grad<const D: usize>(
                tensor: &Self::Tensor<D>,
                grads: &Gradients,
            ) -> Option<<$backend as Backend>::Tensor<D>> {
                grads.wrt(tensor).map(|grad| grad.clone())
            }
        }
    };
}

#[cfg(feature = "ndarray")]
define_impl!(
    name: ADBackendNdArray,
    backend: crate::tensor::backend::ndarray::NdArrayBackend<E>
);
#[cfg(feature = "tch")]
define_impl!(
    name: ADBackendTch,
    backend: crate::tensor::backend::tch::TchBackend<E>
);
