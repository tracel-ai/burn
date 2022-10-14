use crate::backend::autodiff::ADBackendDecorator;
use crate::backend::Backend;
use crate::ops::TensorOpsDevice;
use crate::{execute_ops, register_ops};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};

register_ops!(
    ops UnaryOps,
    name ADTensorDeviceOps state B::Device,
    partial |
        device: &B::Device,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
    | {
        state.output.grad().to_device(*device)
    },
);

impl<B: Backend, const D: usize> TensorOpsDevice<ADBackendDecorator<B>, D>
    for <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>
{
    fn device(&self) -> <ADBackendDecorator<B> as Backend>::Device {
        TensorOpsDevice::device(&self.tensor())
    }

    fn to_device(&self, device: <ADBackendDecorator<B> as Backend>::Device) -> ADTensor<D, B> {
        let tensor = self.tensor();
        execute_ops!(
            input self.node.clone(),
            out TensorOpsDevice::to_device(&tensor, device),
            ops ADTensorDeviceOps::<B, D>::new(tensor.device()),
        )
    }
}
