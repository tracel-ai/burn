use crate::{
    checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    grads::Gradients,
    runtime::AutodiffClient,
    tensor::AutodiffTensor,
    AutodiffBridge,
};
use burn_common::sync_type::SyncType;
use burn_tensor::backend::{AutodiffBackend, Backend};
use core::marker::PhantomData;

/// Enable auto-differentiation on a backend.
///
/// This works as a backend decorator, extending the functionality of any backend with
/// backpropagation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Autodiff<B, C = NoCheckpointing> {
    _b: PhantomData<B>,
    _checkpoint_strategy: PhantomData<C>,
}

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    type Device = B::Device;

    type FullPrecisionBridge = AutodiffBridge<B::FullPrecisionBridge>;

    type FloatTensorPrimitive<const D: usize> = AutodiffTensor<B, D>;
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

    fn sync(device: &B::Device, sync_type: SyncType) {
        B::sync(device, sync_type)
    }
}

impl<B: Backend, C: CheckpointStrategy> AutodiffBackend for Autodiff<B, C> {
    type InnerBackend = B;
    type Gradients = Gradients;

    fn backward<const D: usize>(tensor: AutodiffTensor<B, D>) -> Gradients {
        let client = tensor.node.client.clone();

        AutodiffClient::backward(&client, tensor)
    }

    fn grad<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &Gradients,
    ) -> Option<B::FloatTensorPrimitive<D>> {
        grads.get(tensor)
    }

    fn grad_remove<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive<D>> {
        grads.remove(tensor)
    }
    fn inner<const D: usize>(tensor: AutodiffTensor<B, D>) -> B::FloatTensorPrimitive<D> {
        tensor.primitive
    }

    fn from_inner<const D: usize>(tensor: B::FloatTensorPrimitive<D>) -> AutodiffTensor<B, D> {
        AutodiffTensor::new(tensor)
    }

    fn grad_replace<const D: usize>(
        tensor: &AutodiffTensor<B, D>,
        grads: &mut Self::Gradients,
        grad: B::FloatTensorPrimitive<D>,
    ) {
        grads.remove(tensor);
        grads.register::<B, D>(tensor.node.id, grad);
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
