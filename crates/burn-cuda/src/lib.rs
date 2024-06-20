#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub mod compiler;
pub use device::*;

use burn_jit::JitBackend;
pub use runtime::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = JitBackend<CudaRuntime, F, I>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<JitBackend<CudaRuntime, F, I>>;

#[cfg(test)]
mod tests {
    use super::*;

    pub type TestRuntime = crate::CudaRuntime;

    burn_jit::testgen_all!();
    burn_cube::testgen_all!();
}
