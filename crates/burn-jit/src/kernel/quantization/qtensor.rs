#![allow(missing_docs)] // cube derive macros

use burn_tensor::quantization::QuantizationScheme;
use cubecl::prelude::*;

/// Quantization parameters.
#[derive(CubeLaunch)]
pub struct QParams {
    #[cube(comptime)]
    scheme: QuantizationScheme,
}

/// Quantized tensor representation.
pub type QTensor = Array<Line<u32>>;

#[cube]
impl QParams {
    /// Create a new quantization parameters instance.
    pub fn new(scheme: QuantizationScheme) -> Self {
        QParams { scheme }
    }

    /// Get the floating-point scaling factor.
    pub fn scale(&self, tensor: &QTensor) -> f32 {
        let len = tensor.len();
        match comptime!(self.scheme) {
            QuantizationScheme::PerTensorAffine(_) => match comptime!(tensor.line_size()) {
                // For line size of 1, scale is the last value in the buffer
                1 => f32::bitcast_from(tensor[len - 1][tensor.line_size() - 1]),
                // For any other line size > 1, scale and zero-point offset are the first two elements of the last line
                _ => f32::bitcast_from(tensor[len - 1][tensor.line_size() - 2]),
            },
            // Symmetric quantization only contains the scaling factor as the last element
            QuantizationScheme::PerTensorSymmetric(_) => {
                f32::bitcast_from(tensor[len - 1][tensor.line_size() - 1])
            }
        }
    }

    /// Get the zero-point offset.
    pub fn offset(&self, tensor: &QTensor) -> i32 {
        let len = tensor.len();
        let line_size = comptime!(tensor.line_size());
        match comptime!(self.scheme) {
            QuantizationScheme::PerTensorAffine(_) => match line_size {
                // For line size of 1, the zero-point offset is the penultimate value in the buffer
                1 => i32::cast_from(tensor[len - 2][line_size]),
                // For any other line size > 1, scale and zero-point offset are the first two elements of the last line
                _ => i32::cast_from(tensor[len - 1][line_size]),
            },
            // Symmetric quantization only contains the scaling factor, so we return 0 for the zero-point offset
            QuantizationScheme::PerTensorSymmetric(_) => 0,
        }
    }
}
