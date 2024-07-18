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
    // TODO: dynamic quant? I think we won't support fully static (with observers to record the values on data samples)
    // just yet so this is not required.
    // /// Dynamic quantization computes the quantized parameters at runtime.
    // pub dynamic: bool,
}

impl<B: Backend, C: Calibration> ModuleMapper<B> for Quantizer<C> {
    fn map_float<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let range = self.calibration.compute_range(&tensor);
        let qparams = self.scheme.compute_q_params(range);
        tensor.quantize(&self.scheme, qparams)
    }
}

// /// Describes how to quantize a module by providing quantizer settings for activations and weights respectively.
// pub struct QuantizationConfig<CW: Calibration> {
//     // TODO: quantization config
//     /// The quantizer used to quantize the activations (i.e., a layer's output).
//     // pub activations: Quantizer<CA>,
//     /// The quantizer used to quantize the weights.
//     pub weights: Quantizer<CW>,
// }
