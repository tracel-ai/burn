#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

#[cfg(feature = "template")]
pub use burn_jit::{
    kernel::{into_contiguous, Kernel},
    kernel_source,
    template::{build_info, KernelSource, SourceKernel, SourceTemplate},
};

pub use burn_jit::{tensor::JitTensor, JitBackend};
pub use burn_jit::{FloatElement, IntElement};
pub use cubecl::ir::CubeDim;
pub use cubecl::wgpu::*;

pub type Wgsl = cubecl::wgpu::WgslCompiler;
#[cfg(feature = "spirv")]
pub type SpirV = cubecl::wgpu::spirv::VkSpirvCompiler;

#[cfg(feature = "spirv")]
type Compiler = SpirV;
#[cfg(not(feature = "spirv"))]
type Compiler = Wgsl;

#[cfg(feature = "fusion")]
/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// To configure the wgpu backend, eg. to select what graphics API to use or what memory strategy to use,
/// you have to manually initialize the runtime. For example:
///
/// ```rust, ignore
/// fn custom_init() {
///     let device = Default::default();
///     burn::backend::wgpu::init_sync::<burn::backend::wgpu::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_existing_device`.
///
/// # Notes
///
/// This version of the wgpu backend uses [burn_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Wgpu<F = f32, I = i32, C = Compiler> =
    burn_fusion::Fusion<JitBackend<cubecl::wgpu::WgpuRuntime<C>, F, I>>;

#[cfg(not(feature = "fusion"))]
/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::Dx12) on Windows.
///   - [Metal] on Apple hardware.
///   - [WebGPU](crate::WebGpu) on supported browsers and `wasm` runtimes.
///
/// To configure the wgpu backend, eg. to select what graphics API to use or what memory strategy to use,
/// you have to manually initialize the runtime. For example:
///
/// ```rust, ignore
/// fn custom_init() {
///     let device = Default::default();
///     burn::backend::wgpu::init_sync::<burn::backend::wgpu::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_existing_device`.
///
/// # Notes
///
/// This version of the wgpu backend doesn't use [burn_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Wgpu<F = f32, I = i32, C = Compiler> = JitBackend<cubecl::wgpu::WgpuRuntime<C>, F, I>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;
    pub type TestRuntime = cubecl::wgpu::WgpuRuntime<super::Compiler>;

    burn_jit::testgen_all!();
}
