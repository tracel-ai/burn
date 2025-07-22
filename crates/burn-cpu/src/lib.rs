#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cpu::CpuDevice;
use cubecl::cpu::CpuRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cpu<F = f32, I = i32> = CubeBackend<CpuRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cpu<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CpuRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use burn_cubecl::CubeBackend;

    pub type TestRuntime = cubecl::cpu::CpuRuntime;

    // TODO: Add tests for bf16
    // burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    burn_cubecl::testgen_all!([f32], [i32], [u32]);
}
