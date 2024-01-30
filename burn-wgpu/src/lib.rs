#![warn(missing_docs)]

//! Burn WGPU Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

/// Compute related module.
pub mod compute;
/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

pub(crate) mod codegen;
pub(crate) mod tune;

mod element;
pub use element::{FloatElement, IntElement};

mod device;
pub use device::*;

mod backend;
pub use backend::*;

mod graphics;
pub use graphics::*;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(feature = "fusion")]
/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan](crate::Vulkan) on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal](crate::Metal) on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the [wgpu] backend uses [burn_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Wgpu<G = AutoGraphicsApi, F = f32, I = i32> = burn_fusion::Fusion<WgpuBackend<G, F, I>>;

#[cfg(not(feature = "fusion"))]
/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan](crate::Vulkan) on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal](crate::Metal) on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the [wgpu] backend doesn't uses [burn_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Wgpu<G = AutoGraphicsApi, F = f32, I = i32> = WgpuBackend<G, F, I>;

#[cfg(test)]
mod tests {
    use super::*;

    pub type TestBackend = WgpuBackend;
    pub type ReferenceBackend = burn_ndarray::NdArray<f32>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
