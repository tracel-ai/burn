pub use super::*;

mod quantize_dequantize;
mod reshape;
mod slice;

fn supports_native() -> bool {
    let name = format!("{:?}", burn_tensor::Device::default());
    // TODO: Proper checks for i8 support.
    name.contains("Cuda")
        || name.contains("Rocm")
        || name.contains("Vulkan")
        || name.contains("Metal")
}
