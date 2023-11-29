use crate::{grads::Gradients, graph::backward::backward, tensor::AutodiffTensor};
use burn_tensor::backend::{AutodiffBackend, Backend};
use core::marker::PhantomData;

/// Enable auto-differentiation on a backend.
///
/// This works as a backend decorator, extending the functionality of any backend with
/// backpropagation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B> {
    _b: PhantomData<B>,
}

impl<B: Backend> Backend for Autodiff<B> {
    type Device = B::Device;

    type FullPrecisionElem = B::FullPrecisionElem;
    type FullPrecisionBackend = Autodiff<B::FullPrecisionBackend>;

    type TensorPrimitive<const D: usize> = AutodiffTensor<B, D>;
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

    fn sync(device: &B::Device) {
        B::sync(device);
    }
}

impl<B: Backend> AutodiffBackend for Autodiff<B> {
    type InnerBackend = B;
    type Gradients = Gradients;

    fn backward<const D: usize>(tensor: AutodiffTensor<B, D>) -> Gradients {
        backward(tensor)
    }

    fn grad<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.get(tensor)
    }

    fn grad_remove<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &mut Gradients,
    ) -> Option<B::TensorPrimitive<D>> {
        grads.remove(tensor)
    }
    fn inner<const D: usize>(tensor: AutodiffTensor<B, D>) -> B::TensorPrimitive<D> {
        tensor.primitive
    }

    fn from_inner<const D: usize>(tensor: B::TensorPrimitive<D>) -> AutodiffTensor<B, D> {
        AutodiffTensor::new(tensor)
    }

    fn grad_replace<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &mut Self::Gradients,
        grad: B::TensorPrimitive<D>,
    ) {
        grads.remove(tensor);
        grads.register::<B, D>(tensor.node.clone(), grad);
    }

    fn int_inner<const D: usize>(
        tensor: burn_tensor::ops::IntTensor<Self, D>,
    ) -> burn_tensor::ops::IntTensor<Self::InnerBackend, D> {
        tensor
    }

    fn bool_inner<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self::InnerBackend, D> {
        tensor
    }

    fn int_from_inner<const D: usize>(
        tensor: burn_tensor::ops::IntTensor<Self::InnerBackend, D>,
    ) -> burn_tensor::ops::IntTensor<Self, D> {
        tensor
    }

    fn bool_from_inner<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<Self::InnerBackend, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        tensor
    }
}
