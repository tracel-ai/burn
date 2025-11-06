#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Rocm<F = f32, I = i32, B = u8> = CubeBackend<HipRuntime, F, I, B>;

#[cfg(feature = "fusion")]
pub type Rocm<F = f32, I = i32, B = u8> = burn_fusion::Fusion<CubeBackend<HipRuntime, F, I, B>>;

#[cfg(test)]
mod tests {
    use burn_cubecl::CubeBackend;

    pub type TestRuntime = cubecl::hip::HipRuntime;
    use half::f16;

    // TODO: Add tests for bf16
    // burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    burn_cubecl::testgen_all!([f16, f32], [i32], [u32]);
}
