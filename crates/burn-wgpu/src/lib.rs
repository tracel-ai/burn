#[macro_use]
extern crate derive_new;

extern crate alloc;

mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

#[cfg(feature = "template")]
pub use burn_jit::{
    compute::Kernel,
    gpu::WorkgroupSize,
    kernel::{into_contiguous, GpuComputeShaderPhase},
    kernel_wgsl,
    template::{build_info, KernelSource, SourceKernel, SourceTemplate},
};

pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

pub use burn_jit::compute::WorkGroup;
pub use burn_jit::{tensor::JitTensor, JitBackend};

pub use crate::compute::WebGPUApi;

#[cfg(feature = "dawn")]
pub use crate::compute::DawnApi;
#[cfg(feature = "wgpu")]
pub use crate::compute::WgpuApi;

#[cfg(all(feature = "fusion", feature = "wgpu"))]
/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the [wgpu] backend uses [burn_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Wgpu<G = AutoGraphicsApi, F = f32, I = i32> =
    burn_fusion::Fusion<JitBackend<WgpuRuntime<WgpuApi, G>, F, I>>;

#[cfg(all(not(feature = "fusion"), feature = "wgpu"))]
/// Tensor backend that uses the [wgpu] crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the [wgpu] backend doesn't use [burn_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Wgpu<G = AutoGraphicsApi, F = f32, I = i32> = JitBackend<WgpuRuntime<WgpuApi, G>, F, I>;

#[cfg(all(feature = "fusion", feature = "dawn"))]
/// Tensor backend that uses Dawn for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the Dawn backend uses [burn_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Dawn<G = AutoGraphicsApi, F = f32, I = i32> =
    burn_fusion::Fusion<JitBackend<WgpuRuntime<DawnApi, G>, F, I>>;

#[cfg(all(not(feature = "fusion"), feature = "dawn"))]
/// Tensor backend that uses Dawn for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// # Notes
///
/// This version of the Dawn backend doesn't use [burn_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Dawn<G = AutoGraphicsApi, F = f32, I = i32> = JitBackend<WgpuRuntime<DawnApi, G>, F, I>;

#[cfg(all(test, feature = "wgpu"))]
mod tests_wgpu {
    use super::*;

    pub type TestRuntime = crate::WgpuRuntime<WgpuApi, AutoGraphicsApi>;

    burn_jit::testgen_all!();
}

#[cfg(all(test, feature = "dawn"))]
mod tests_dawn {
    use super::*;

    pub type TestRuntime = crate::WgpuRuntime<DawnApi, AutoGraphicsApi>;

    burn_jit::testgen_all!();
}
