#![allow(missing_docs)] // cube derive macros

use burn_tensor::quantization::{QuantizationMode, QuantizationScheme};
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

    /// Get the quantization parameters values.
    pub fn values(&self, tensor: &QTensor) -> (f32, i32) {
        let len = tensor.len();
        match comptime!(self.scheme) {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, _) => {
                match comptime!(tensor.line_size()) {
                    // For line size of 1, scale is the last value in the buffer
                    1 => (
                        f32::bitcast_from(tensor[len - 1][tensor.line_size() - 1]),
                        i32::cast_from(tensor[len - 2][tensor.line_size() - 1]),
                    ),
                    // For any other line size > 1, scale and zero-point offset are the last two elements
                    _ => {
                        let values = tensor[len - 1];
                        (
                            f32::bitcast_from(values[tensor.line_size() - 1]),
                            i32::cast_from(values[tensor.line_size() - 2]),
                        )
                    }
                }
            }
            // Symmetric quantization only contains the scaling factor as the last element
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, _) => (
                f32::bitcast_from(tensor[len - 1][tensor.line_size() - 1]),
                0,
            ),
            QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
        }
    }
}
