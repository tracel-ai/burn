#![allow(missing_docs)] // cube derive macros

use burn_tensor::quantization::{QuantInputType, QuantLevel, QuantMode, QuantScheme};
use cubecl::prelude::*;

/// Quantization parameters.
#[derive(CubeLaunch, CubeType)]
pub struct QParams {
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    pub num_quants: u32,
}

#[cube]
impl QParams {
    /// Create a new quantization parameters instance.
    pub fn new(#[comptime] scheme: QuantScheme) -> Self {
        let num_quants = comptime!((scheme.size_bits_stored() / scheme.q_type.size_bits()) as u32);
        QParams { scheme, num_quants }
    }

    /// Get the quantization parameters values.
    pub fn scale<F: Float>(&self, scale_tensor: &Tensor<F>, in_pos: u32) -> F {
        match comptime!(self.scheme) {
            // Symmetric quantization only contains the scaling factor as the last element
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => scale_tensor[0],
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => {
                // The input position is `num_quants` smaller because it acts as vectorize with a line
                // size, but the scales don't have any line size.
                let position = in_pos * self.num_quants;
                scale_tensor[position / comptime! {block_size as u32}]
            }
        }
    }
}
