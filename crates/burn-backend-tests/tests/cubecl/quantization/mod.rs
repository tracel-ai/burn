pub use super::*;

mod quantize_dequantize;
mod reshape;

fn supports_native() -> bool {
    let name = <TestBackend as burn_tensor::backend::Backend>::name(&Default::default());
    // TODO: Proper checks for i8 support.
    name.contains("cuda")
        || name.contains("rocm")
        || name.contains("hip")
        || name.contains("vulkan")
        || name.contains("spirv")
        || name.contains("metal")
        || name.contains("msl")
}
