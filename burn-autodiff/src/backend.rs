use crate::{grads::Gradients, graph::backward::backward, tensor::ADTensor};
use burn_tensor::backend::{ADBackend, Backend};

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
    type TensorPrimitive<const D: usize> = ADTensor<B, D>;
    type BoolTensorPrimitive<const D: usize> = B::BoolTensorPrimitive<D>;

    fn ad_enabled() -> bool {
        true
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
    type Gradients = Gradients;

    fn backward<const D: usize>(tensor: &ADTensor<B, D>) -> Gradients {
        backward(tensor.clone())
    }

    fn grad<const D: usize>(
        tensor: &ADTensor<B, D>,
        grads: &Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.get(tensor)
    }

    fn inner<const D: usize>(tensor: &ADTensor<B, D>) -> B::TensorPrimitive<D> {
        tensor.primitive.clone()
    }

    fn from_inner<const D: usize>(tensor: B::TensorPrimitive<D>) -> ADTensor<B, D> {
        ADTensor::new(tensor)
    }
}
