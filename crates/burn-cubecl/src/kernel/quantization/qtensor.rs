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
        let num_quants = comptime!(
            let size_quant = match scheme.q_type {
                QuantInputType::QInt8 => 8u32,
            };
            let size_store = match scheme.q_store_type {
                burn_tensor::quantization::QuantStoreType::Native => size_quant,
                burn_tensor::quantization::QuantStoreType::I8 => 8u32,
                burn_tensor::quantization::QuantStoreType::I32 => 32u32,
            };

            size_store / size_quant
        );
        QParams { scheme, num_quants }
    }

    /// Get the quantization parameters values.
    pub fn scale(&self, scale_tensor: &Tensor<f32>, in_pos: u32) -> f32 {
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
                // Since the input position is num quants smaller because it acks as vectorize with a line
                // size, but the scales don't have any line size.
                let position = in_pos * self.num_quants;
                scale_tensor[position / comptime! {block_size as u32}]
            }
        }
    }
}
