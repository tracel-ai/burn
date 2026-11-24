use crate::OptimizerStateMapper;
use burn_core::tensor::{Device, Tensor};

pub(crate) fn move_state<S: crate::StateMappable>(state: S, device: &Device) -> S {
    state.map_tensors(&DeviceOptimizerStateMapper { device })
}

struct DeviceOptimizerStateMapper<'a> {
    device: &'a Device,
}
impl OptimizerStateMapper for DeviceOptimizerStateMapper<'_> {
    fn map<const D: usize>(&self, tensor: Tensor<D>, _path: &crate::StatePath<'_>) -> Tensor<D> {
        tensor.to_device(self.device)
    }
}
