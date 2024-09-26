use crate::{
    checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    grads::Gradients,
    runtime::AutodiffClient,
    tensor::AutodiffTensor,
    AutodiffBridge,
};
use burn_common::sync_type::SyncType;
use burn_tensor::{
    backend::{AutodiffBackend, Backend},
    ops::{BoolTensor, IntTensor, QuantizedTensor},
};
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

    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;

    type BoolTensorPrimitive = B::BoolTensorPrimitive;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;
    type QuantizedEncoding = B::QuantizedEncoding;

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

    fn backward(tensor: AutodiffTensor<B>) -> Gradients {
        let client = tensor.node.client.clone();

        AutodiffClient::backward::<B>(&client, tensor)
    }

    fn grad(tensor: &AutodiffTensor<B>, grads: &Gradients) -> Option<B::FloatTensorPrimitive> {
        grads.get::<B>(tensor)
    }

    fn grad_remove(
        tensor: &AutodiffTensor<B>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive> {
        grads.remove::<B>(tensor)
    }
    fn inner(tensor: AutodiffTensor<B>) -> B::FloatTensorPrimitive {
        tensor.primitive
    }

    fn from_inner(tensor: B::FloatTensorPrimitive) -> AutodiffTensor<B> {
        AutodiffTensor::new(tensor)
    }

    fn grad_replace(
        tensor: &AutodiffTensor<B>,
        grads: &mut Self::Gradients,
        grad: B::FloatTensorPrimitive,
    ) {
        grads.remove::<B>(tensor);
        grads.register::<B>(tensor.node.id, grad);
    }

    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend> {
        tensor
    }

    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend> {
        tensor
    }

    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self> {
        tensor
    }

    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self> {
        tensor
    }

    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend> {
        tensor
    }

    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self> {
        tensor
    }
}
