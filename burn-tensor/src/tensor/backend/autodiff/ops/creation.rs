use crate::tensor::{
    backend::{autodiff::ADTensor, Backend},
    ops::*,
    Data, Distribution, Element, Shape,
};
use rand::distributions::Standard;

impl<B: Backend, const D: usize> TensorCreationLike<B::Elem, D> for ADTensor<D, B>
where
    Standard: rand::distributions::Distribution<B::Elem>,
{
    fn new_like_empty(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_empty())
    }

    fn new_like_random(&self, distribution: Distribution<B::Elem>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_random(distribution))
    }

    fn new_like_data(&self, data: Data<B::Elem, D>) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_data(data))
    }

    fn new_like_zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_zeros())
    }

    fn new_like_ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().new_like_ones())
    }
}

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty
    ) => {
        impl<E: Element, const D: usize> TensorCreationFork<$backend, D>
            for ADTensor<D, $backend_inner>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            fn new_fork_empty<const D2: usize>(
                &self,
                shape: Shape<D2>,
            ) -> ADTensor<D2, $backend_inner> {
                ADTensor::from_tensor(self.tensor().new_fork_empty(shape))
            }

            fn new_fork_random<const D2: usize>(
                &self,
                shape: Shape<D2>,
                distribution: Distribution<E>,
            ) -> ADTensor<D2, $backend_inner> {
                ADTensor::from_tensor(self.tensor().new_fork_random(shape, distribution))
            }

            fn new_fork_data<const D2: usize>(
                &self,
                data: Data<E, D2>,
            ) -> ADTensor<D2, $backend_inner> {
                ADTensor::from_tensor(self.tensor().new_fork_data(data))
            }

            fn new_fork_zeros<const D2: usize>(
                &self,
                shape: Shape<D2>,
            ) -> ADTensor<D2, $backend_inner> {
                ADTensor::from_tensor(self.tensor().new_fork_zeros(shape))
            }

            fn new_fork_ones<const D2: usize>(
                &self,
                shape: Shape<D2>,
            ) -> ADTensor<D2, $backend_inner> {
                ADTensor::from_tensor(self.tensor().new_fork_ones(shape))
            }
        }
    };
}

#[cfg(feature = "ndarray")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendNdArray::<E>,
    crate::tensor::backend::ndarray::NdArrayBackend::<E>
);

#[cfg(feature = "tch")]
define_impl!(
    crate::tensor::backend::autodiff::ADBackendTch::<E>,
    crate::tensor::backend::tch::TchBackend::<E>
);

impl<B: Backend, const D: usize> Zeros<Self> for ADTensor<D, B> {
    fn zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().zeros())
    }
}

impl<B: Backend, const D: usize> Ones<Self> for ADTensor<D, B> {
    fn ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().ones())
    }
}
