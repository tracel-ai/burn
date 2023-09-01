use crate::{grads::Gradients, graph::backward::backward, tensor::ADTensor};
use burn_tensor::backend::{ADBackend, Backend};

/// A decorator for a backend that enables automatic differentiation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ADBackendDecorator<B> {
    _b: B,
}

impl<B: Backend> Backend for ADBackendDecorator<B> {
    type Device = B::Device;

    type FullPrecisionElem = B::FullPrecisionElem;
    type FullPrecisionBackend = ADBackendDecorator<B::FullPrecisionBackend>;

    type TensorPrimitive<const D: usize> = ADTensor<B, D>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = B::IntTensorPrimitive<D>;
    type IntElem = B::IntElem;

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

    fn backward<const D: usize>(tensor: ADTensor<B, D>) -> Gradients {
        backward(tensor)
    }

    fn grad<const D: usize>(
        tensor: &ADTensor<B, D>,
        grads: &Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.get(tensor)
    }

    fn grad_remove<const D: usize>(
        tensor: &ADTensor<B, D>,
        grads: &mut Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.remove(tensor)
    }
    fn inner<const D: usize>(tensor: ADTensor<B, D>) -> B::TensorPrimitive<D> {
        tensor.primitive
    }

    fn from_inner<const D: usize>(tensor: B::TensorPrimitive<D>) -> ADTensor<B, D> {
        ADTensor::new(tensor)
    }

    fn grad_replace<const D: usize>(
        tensor: &ADTensor<B, D>,
        grads: &mut Self::Gradients,
        grad: B::TensorPrimitive<D>,
    ) {
        grads.remove(tensor);
        grads.register::<B, D>(tensor.node.clone(), grad);
    }
}
