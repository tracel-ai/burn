use burn_tensor::{
    backend::Backend,
    quantization::{Calibration, QuantizationScheme},
    Tensor,
};

use crate::module::{ModuleMapper, ParamId};

/// Describes how to quantize a module.
pub struct Quantizer<C: Calibration> {
    /// The calibration method used in quantization.
    pub calibration: C,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
}

impl<B: Backend, C: Calibration> ModuleMapper<B> for Quantizer<C> {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let range = self.calibration.compute_range(&tensor);
        let qparams = self.scheme.compute_q_params(range);
        tensor.quantize(&self.scheme, qparams)
    }
}
