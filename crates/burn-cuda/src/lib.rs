extern crate alloc;

use burn_jit::JitBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;
use half::f16;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f16, I = i32> = JitBackend<CudaRuntime, F, I>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f16, I = i32> = burn_fusion::Fusion<JitBackend<CudaRuntime, F, I>>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;

    pub type TestRuntime = cubecl::cuda::CudaRuntime;

    burn_jit::testgen_all!();
}
