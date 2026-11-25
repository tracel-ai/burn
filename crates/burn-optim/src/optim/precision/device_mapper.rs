use crate::{OptimizerStateMapper, StatePath};
use burn_core::tensor::{Bool, Device, Int, Tensor};

pub(crate) fn move_state<S: crate::StateMappable>(state: S, device: &Device) -> S {
    state.map_tensors(&DeviceOptimizerStateMapper { device })
}

struct DeviceOptimizerStateMapper<'a> {
    device: &'a Device,
}
impl OptimizerStateMapper for DeviceOptimizerStateMapper<'_> {
    fn map<const D: usize>(&self, tensor: Tensor<D>, _path: &StatePath<'_>) -> Tensor<D> {
        tensor.to_device(self.device)
    }

    fn map_int<const D: usize>(&self, tensor: Tensor<D, Int>, _: &StatePath<'_>) -> Tensor<D, Int> {
        tensor.to_device(self.device)
    }

    fn map_bool<const D: usize>(
        &self,
        tensor: Tensor<D, Bool>,
        _: &StatePath<'_>,
    ) -> Tensor<D, Bool> {
        tensor.to_device(self.device)
    }
}
