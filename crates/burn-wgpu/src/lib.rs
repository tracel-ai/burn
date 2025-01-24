#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

#[cfg(feature = "template")]
pub use burn_jit::{
    kernel::{into_contiguous, Kernel},
    kernel_source,
    template::{build_info, KernelSource, SourceKernel, SourceTemplate},
};

pub use burn_jit::{tensor::JitTensor, JitBackend};
pub use burn_jit::{BoolElement, FloatElement, IntElement};
pub use cubecl::flex32;
pub use cubecl::CubeDim;

pub use cubecl::wgpu::{
    init_device, init_setup, init_setup_async, MemoryConfiguration, RuntimeOptions, WgpuDevice,
    WgpuResource, WgpuRuntime, WgpuSetup, WgpuStorage,
};
// Vulkan and WebGpu would have conflicting type names
pub mod graphics {
    pub use cubecl::wgpu::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan, WebGpu};
}

#[cfg(feature = "cubecl-spirv")]
pub use cubecl::wgpu::spirv::SpirvCompiler;
#[cfg(feature = "cubecl-wgsl")]
pub use cubecl::wgpu::WgslCompiler;

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
///     burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_device`.
///
/// # Notes
///
/// This version of the wgpu backend uses [burn_fusion] to compile and optimize streams of tensor
/// operations for improved performance.
///
/// You can disable the `fusion` feature flag to remove that functionality, which might be
/// necessary on `wasm` for now.
pub type Wgpu<F = f32, I = i32, B = u32, C = cubecl::wgpu::WgslCompiler> =
    burn_fusion::Fusion<JitBackend<cubecl::wgpu::WgpuRuntime<C>, F, I, B>>;

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
///     burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
///         &device,
///         Default::default(),
///     );
/// }
/// ```
/// will mean the given device (in this case the default) will be initialized to use Vulkan as the graphics API.
/// It's also possible to use an existing wgpu device, by using `init_device`.
///
/// # Notes
///
/// This version of the wgpu backend doesn't use [burn_fusion] to compile and optimize streams of tensor
/// operations.
///
/// You can enable the `fusion` feature flag to add that functionality, which might improve
/// performance.
pub type Wgpu<F = f32, I = i32, B = u32, C = cubecl::wgpu::WgslCompiler> =
    JitBackend<cubecl::wgpu::WgpuRuntime<C>, F, I, B>;

#[cfg(feature = "vulkan")]
/// Tensor backend that leverages the Vulkan graphics API to execute GPU compute shaders compiled to SPIR-V.
pub type Vulkan<F = f32, I = i32, B = u8> = Wgpu<F, I, B, cubecl::wgpu::spirv::VkSpirvCompiler>;

#[cfg(feature = "webgpu")]
/// Tensor backend that uses the wgpu crate to execute GPU compute shaders written in WGSL.
pub type WebGpu<F = f32, I = i32, B = u32> = Wgpu<F, I, B, WgslCompiler>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;
    #[cfg(feature = "vulkan")]
    pub use half::f16;

    #[cfg(feature = "cubecl-spirv")]
    type Compiler = cubecl::wgpu::spirv::VkSpirvCompiler;
    #[cfg(not(feature = "cubecl-spirv"))]
    type Compiler = cubecl::wgpu::WgslCompiler;
    pub type TestRuntime = cubecl::wgpu::WgpuRuntime<Compiler>;

    // Don't test `flex32` for now, burn sees it as `f32` but is actually `f16` precision, so it
    // breaks a lot of tests from precision issues
    #[cfg(feature = "vulkan")]
    burn_jit::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    #[cfg(not(feature = "vulkan"))]
    burn_jit::testgen_all!([f32], [i32], [u32]);
}
