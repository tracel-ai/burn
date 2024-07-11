use burn_tensor::{backend::Backend, Tensor};

use crate::module::{ModuleMapper, ParamId};

use super::Calibration;

/// Describes how to quantize a module by providing quantizer settings for activations and weights respectively.
pub struct QuantizationConfig<CW: Calibration> {
    // TODO:
    /// The quantizer used to quantize the activations (i.e., a layer's output).
    // pub activations: Quantizer<CA>,
    /// The quantizer used to quantize the weights.
    pub weights: Quantizer<CW>,
}

/// Describes how to quantize a module.
pub struct Quantizer<C: Calibration> {
    /// The calibration method used in quantization.
    pub calibration: C,
    // TODO: dynamic quant
    // /// Dynamic quantization computes the quantized parameters at runtime.
    // pub dynamic: bool,
}

impl<B: Backend, C: Calibration> ModuleMapper<B> for Quantizer<C> {
    fn map_float<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let strategy = self.calibration.configure(&tensor);
        tensor.quantize(strategy)
    }
}
