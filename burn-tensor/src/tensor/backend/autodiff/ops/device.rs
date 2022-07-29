use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::ForwardUnaryRecordedOps;
use crate::graph::ops::{UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::{ADKind, ADTensor};
use crate::tensor::TensorType;
use crate::tensor::{backend::autodiff::ADBackend, ops::TensorOpsDevice, Backend, Element, Tensor};
use rand::distributions::Standard;
use std::sync::Arc;

#[derive(Debug)]
struct ADTensorOpsDevice<P, const D1: usize, B: Backend> {
    device: <B as Backend>::Device,
    _kind: ADKind<P>,
}

impl<P: Default, const D1: usize, B: Backend> ADTensorOpsDevice<P, D1, B> {
    pub fn new(device: <B as Backend>::Device) -> Self {
        Self {
            device,
            _kind: ADKind::new(),
        }
    }
}

impl<P, const D1: usize, B> UnaryOps<Tensor<D1, B>, Tensor<D1, B>> for ADTensorOpsDevice<P, D1, B>
where
    B: Backend<E = P> + TensorType<D1, B>,
    P: Element,
    Standard: rand::distributions::Distribution<P>,
{
    fn partial(&self, state: &UnaryOpsNodeState<Tensor<D1, B>, Tensor<D1, B>>) -> Tensor<D1, B> {
        let tensor = state.output.grad();
        tensor.to_device(self.device)
    }
}

macro_rules! define_impl {
    ($b:ty) => {
        impl<E, const D: usize> TensorOpsDevice<E, D, ADBackend<E, $b>>
            for ADTensor<E, D, Tensor<D, $b>>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            fn device(&self) -> <$b as Backend>::Device {
                let tensor = self.tensor();
                tensor.device()
            }

            fn to_device(&self, device: <$b as Backend>::Device) -> Self {
                let input = self.tensor();
                let out = TensorOpsDevice::to_device(&input, device.clone());

                let state = ForwardNodeState::new(out);

                let ops = ADTensorOpsDevice::<E, D, $b>::new(device.clone());
                let ops = Arc::new(ops);
                let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
                let ops = Arc::new(ops);

                let node = ForwardNode::from_unary(&self.node, state, ops);
                let node = Arc::new(node);

                let shape = self.shape.clone();
                let kind = self.kind.clone();

                ADTensor { node, shape, kind }
            }
        }
    };
}

crate::register_ad_backend!();
