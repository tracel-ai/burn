use crate::graph::grad::Grads;
use crate::tensor::ADTensor;
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
    type TensorPrimitive<const D: usize> = ADTensor<D, B>;
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
    type Gradients = Grads;

    fn backward<const D: usize>(tensor: &ADTensor<D, B>) -> Grads {
        tensor.backward()
    }

    fn grad<const D: usize>(
        tensor: &ADTensor<D, B>,
        grads: &Grads,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.wrt(tensor).cloned()
    }

    fn inner<const D: usize>(tensor: &ADTensor<D, B>) -> B::TensorPrimitive<D> {
        tensor.tensor()
    }

    fn from_inner<const D: usize>(tensor: B::TensorPrimitive<D>) -> ADTensor<D, B> {
        ADTensor::from_tensor(tensor)
    }

    fn node_id<const D: usize>(tensor: &Self::TensorPrimitive<D>) -> String {
        tensor.node.id.to_string()
    }
}
