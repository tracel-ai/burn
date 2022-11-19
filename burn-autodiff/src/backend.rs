use crate::graph::grad::Gradients;
use crate::tensor::ADTensor;
use burn_tensor::backend::Backend;
use burn_tensor::{Data, Distribution, Shape};

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

    fn backward<const D: usize>(tensor: &ADTensor<D, B>) -> Gradients {
        tensor.backward()
    }

    fn grad<const D: usize>(
        tensor: &ADTensor<D, B>,
        grads: &Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.wrt(tensor).cloned()
    }

    fn inner<const D: usize>(tensor: &ADTensor<D, B>) -> B::TensorPrimitive<D> {
        tensor.tensor()
    }

    fn from_inner<const D: usize>(tensor: B::TensorPrimitive<D>) -> ADTensor<D, B> {
        ADTensor::from_tensor(tensor)
    }
}

pub(crate) type ADBackendTensorPrimitive<const D: usize, B> =
    <<B as ADBackend>::InnerBackend as Backend>::TensorPrimitive<D>;

pub trait ADBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, Elem = Self::Elem>;

    fn backward<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> Gradients;
    fn grad<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
        grads: &Gradients,
    ) -> Option<ADBackendTensorPrimitive<D, Self>>;
    fn inner<const D: usize>(
        tensor: &Self::TensorPrimitive<D>,
    ) -> <Self::InnerBackend as Backend>::TensorPrimitive<D>;
    fn from_inner<const D: usize>(
        tensor: <Self::InnerBackend as Backend>::TensorPrimitive<D>,
    ) -> Self::TensorPrimitive<D>;
}
