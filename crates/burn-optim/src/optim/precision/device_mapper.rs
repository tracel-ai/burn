use crate::OptimizerStateMapper;
use burn_core::tensor::{Device, Tensor};

pub(crate) struct DeviceOptimizerStateMapper<'a> {
    device: &'a Device,
}
impl<'a> DeviceOptimizerStateMapper<'a> {
    pub(crate) fn new(device: &'a Device) -> Self {
        Self { device }
    }
}
impl OptimizerStateMapper for DeviceOptimizerStateMapper<'_> {
    fn map<const D: usize>(&mut self, tensor: Tensor<D>, _name: &str) -> Tensor<D> {
        tensor.to_device(self.device)
    }
}
