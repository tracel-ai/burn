use burn_tensor::{
    backend::Backend,
    quantization::{Calibration, QuantizationScheme},
    Tensor,
};

use crate::module::{ModuleMapper, ParamId};

/// Describes how to quantize a module.
pub struct Quantizer {
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
}

impl<B: Backend> ModuleMapper<B> for Quantizer {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let range = self.scheme.compute_range(&tensor, &self.calibration);
        let qparams = self.scheme.compute_q_params(range);
        tensor.quantize(&self.scheme, qparams)
    }
}
