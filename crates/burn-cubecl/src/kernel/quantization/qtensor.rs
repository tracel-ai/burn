#![allow(missing_docs)] // cube derive macros

use burn_tensor::quantization::{BlockLayout, QuantizationMode, QuantizationScheme};
use cubecl::prelude::*;

/// Quantization parameters.
#[derive(CubeLaunch, CubeType)]
pub struct QParams {
    #[cube(comptime)]
    scheme: QuantizationScheme,
    #[cube(comptime)]
    num_blocks: u32,
}

/// Quantized tensor representation.
pub type QTensor = Array<Line<u32>>;

#[cube]
impl QParams {
    /// Create a new quantization parameters instance.
    pub fn new(scheme: QuantizationScheme, #[comptime] num_blocks: u32) -> Self {
        QParams { scheme, num_blocks }
    }

    /// Get the quantization parameters values.
    pub fn values(&self, tensor: &QTensor, value_pos: u32) -> (f32, i32) {
        let len = tensor.len();
        match comptime!(self.scheme) {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, _) => {
                match comptime!(tensor.line_size()) {
                    // For line size of 1, scale is the last value in the buffer
                    1 => (
                        f32::reinterpret(tensor[len - 1][0]),
                        i32::cast_from(tensor[len - 2][0]),
                    ),
                    // For any other line size > 1, scale and zero-point offset are the last two elements
                    _ => {
                        let values = tensor[len - 1];
                        (
                            f32::reinterpret(values[tensor.line_size() - 1]),
                            i32::cast_from(values[tensor.line_size() - 2]),
                        )
                    }
                }
            }
            // Symmetric quantization only contains the scaling factor as the last element
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, _) => {
                (f32::reinterpret(tensor[len - 1][tensor.line_size() - 1]), 0)
            }
            // For affine quantization, there are 2 parameters per block
            // The (scale, offset) parameters are stored contiguously by parameter type
            // [offset, offset, offset, ..., scale, scale, scale, ...]
            // (but we might want to store them with each block in the future?)
            QuantizationScheme::PerBlock(
                QuantizationMode::Affine,
                _dtype,
                BlockLayout::Flat(block_size),
            ) => {
                // For each position in the quantized tensor, there are 4 packed values.
                // The block size must be a factor of 4, so at least [4, 8, ...] values are contained in a single block
                let line_size = tensor.line_size();
                let block_idx = value_pos * 4 / block_size;

                let scale =
                    tensor[len - (self.num_blocks - block_idx) / line_size][block_idx % line_size];
                let offset = tensor[len - (2 * self.num_blocks - block_idx / line_size)]
                    [block_idx % line_size];

                (f32::reinterpret(scale), i32::cast_from(offset))
            }
            QuantizationScheme::PerBlock(
                QuantizationMode::Symmetric,
                _dtype,
                BlockLayout::Flat(block_size),
            ) => {
                // For each position in the quantized tensor, there are 4 packed values.
                // The block size must be a factor of 4, so at least [4, 8, ...] values are contained in a single block
                let line_size = tensor.line_size();
                let block_idx = value_pos * 4 / block_size;

                let scale =
                    tensor[len - (self.num_blocks - block_idx) / line_size][block_idx % line_size];
                (f32::reinterpret(scale), 0)
            }
            _ => comptime!(unimplemented!()),
        }
    }
}
