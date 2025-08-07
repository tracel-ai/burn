use burn_tensor::{
    Tensor,
    backend::Backend,
    quantization::{Calibration, QuantScheme, compute_q_params, compute_range},
};

use crate::module::{ModuleMapper, ParamId};

/// Describes how to quantize a module.
pub struct Quantizer {
    /// The calibration method used in quantization.
    pub calibration: Calibration,
    /// The quantization scheme.
    pub scheme: QuantScheme,
}

impl<B: Backend> ModuleMapper<B> for Quantizer {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let range = compute_range(&self.scheme, &tensor, &self.calibration);
        let qparams = compute_q_params(&self.scheme, range);
        tensor.quantize(&self.scheme, qparams)
    }
}
