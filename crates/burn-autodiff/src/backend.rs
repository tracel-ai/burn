use crate::{
    checkpoint::strategy::{CheckpointStrategy, NoCheckpointing},
    grads::Gradients,
    runtime::AutodiffClient,
    tensor::AutodiffTensor,
};
use alloc::{format, string::String};
use burn_tensor::{
    backend::{AutodiffBackend, Backend},
    container::TensorContainerError,
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

    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;

    type BoolTensorPrimitive = B::BoolTensorPrimitive;
    type BoolElem = B::BoolElem;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;
    type QuantizedEncoding = B::QuantizedEncoding;

    fn ad_enabled() -> bool {
        true
    }

    fn name(device: &Self::Device) -> String {
        format!("autodiff<{}>", B::name(device))
    }

    fn seed(seed: u64) {
        B::seed(seed)
    }

    fn sync(device: &B::Device) {
        B::sync(device)
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
        match grads.get::<B>(tensor) {
            Ok(tensor) => Some(tensor),
            Err(error) => match error {
                TensorContainerError::NotFound => None,
                TensorContainerError::DowncastError => panic!(
                    "Downcast mismatch when retrieving tensor. If you are trying to retrieve the gradients for a given parameter id, make sure to use the inner backend. Gradients are not stored on the autodiff backend."
                ),
            },
        }
    }

    fn grad_remove(
        tensor: &AutodiffTensor<B>,
        grads: &mut Gradients,
    ) -> Option<B::FloatTensorPrimitive> {
        match grads.remove::<B>(tensor) {
            Ok(tensor) => Some(tensor),
            Err(error) => match error {
                TensorContainerError::NotFound => None,
                TensorContainerError::DowncastError => panic!(
                    "Downcast mismatch when retrieving tensor. If you are trying to remove the gradients for a given parameter id, make sure to use the inner backend. Gradients are not stored on the autodiff backend."
                ),
            },
        }
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
        if let Err(TensorContainerError::DowncastError) = grads.remove::<B>(tensor) {
            panic!(
                "Downcast mismatch when retrieving tensor. If you are trying to replace the gradients for a given parameter id, make sure to use the inner backend. Gradients are not stored on the autodiff backend."
            )
        };
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
