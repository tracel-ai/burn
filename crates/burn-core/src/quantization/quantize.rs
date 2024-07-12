use burn_tensor::{backend::Backend, Tensor};

use crate::module::{ModuleMapper, ParamId};

use super::Calibration;

/// Describes how to quantize a module.
pub struct Quantizer<C: Calibration> {
    /// The calibration method used in quantization.
    pub calibration: C,
}

impl<B: Backend, C: Calibration> ModuleMapper<B> for Quantizer<C> {
    fn map_float<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let strategy = self.calibration.configure(&tensor);
        tensor.quantize(strategy)
    }
}
