#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

use burn_jit::JitBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = JitBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<JitBackend<CudaRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;

    pub type TestRuntime = cubecl::cuda::CudaRuntime;
    pub use half::{bf16, f16};

    burn_jit::testgen_all!([f16, bf16, f32], [i8, i16, i32, i64], [u8, u32]);
}
