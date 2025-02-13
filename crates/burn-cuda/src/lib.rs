#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use burn_cubecl::CubeBackend;

    pub type TestRuntime = cubecl::cuda::CudaRuntime;
    pub use half::f16;

    // TODO: Add tests for bf16
    burn_cubecl::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
}
