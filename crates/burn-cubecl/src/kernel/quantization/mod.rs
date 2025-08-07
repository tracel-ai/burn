mod dequantize;
mod qtensor;
mod quantize;

pub use dequantize::*;
pub use qtensor::*;
pub use quantize::*;

fn check_block_size_compat(scheme: &burn_tensor::quantization::QuantScheme, div: usize) {
    // Validate block size compatibility
    if let burn_tensor::quantization::QuantScheme {
        level: burn_tensor::quantization::QuantLevel::Block(block_size),
        ..
    } = scheme
    {
        assert!(
            *block_size % div == 0,
            "Block size must be divisible by {div}, got block_size={block_size}"
        );
    }
}
