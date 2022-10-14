use super::ADTensor;
use crate::graph::grad::Gradients;
use crate::tensor::backend::{ADBackend, Backend};
use crate::tensor::{Data, Distribution, Shape};

#[derive(Clone, Copy, Debug, Default)]
pub struct ADBackendDecorator<B> {
    _b: B,
}

impl<B: Backend> Backend for ADBackendDecorator<B> {
    type Device = B::Device;
    type Elem = B::Elem;
    type FullPrecisionElem = B::FullPrecisionElem;
    type IntegerBackend = B::IntegerBackend;
    type FullPrecisionBackend = ADBackendDecorator<B::FullPrecisionBackend>;
    type TensorPrimitive<const D: usize> = ADTensor<D, B>;
    type BoolTensorPrimitive<const D: usize> = B::BoolTensorPrimitive<D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D> {
        let tensor = B::from_data(data, device);
        ADTensor::from_tensor(tensor)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D> {
        B::from_data_bool(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D> {
        Self::from_inner(B::random(shape, distribution, device))
    }

    fn ad_enabled() -> bool {
        true
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_inner(B::zeros(shape, device))
    }

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_inner(B::ones(shape, device))
    }

    fn name() -> String {
        format!("autodiff<{}>", B::name())
    }

    fn seed(seed: u64) {
        B::seed(seed)
    }
}

impl<B: Backend> ADBackend for ADBackendDecorator<B> {
    type InnerBackend = B;

    fn backward<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> Gradients {
        tensor.backward()
    }
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.wrt(tensor).cloned()
    }

    fn inner<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D> {
        tensor.tensor()
    }

    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D> {
        ADTensor::from_tensor(tensor)
    }
}

#[cfg(feature = "ndarray")]
pub type ADBackendNdArray<E> =
    ADBackendDecorator<crate::tensor::backend::ndarray::NdArrayBackend<E>>;

#[cfg(feature = "tch")]
pub type ADBackendTch<E> = ADBackendDecorator<crate::tensor::backend::tch::TchBackend<E>>;
