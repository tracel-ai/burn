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
pub use cubecl::wgpu::*;
pub use cubecl::CubeDim;

pub type Wgsl = cubecl::wgpu::WgslCompiler;
#[cfg(feature = "spirv")]
pub type SpirV = cubecl::wgpu::spirv::VkSpirvCompiler;

#[cfg(feature = "spirv")]
type Compiler = SpirV;
#[cfg(feature = "spirv")]
type Bool = u8;
#[cfg(not(feature = "spirv"))]
type Compiler = Wgsl;
#[cfg(not(feature = "spirv"))]
type Bool = u32;

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
pub type Wgpu<F = f32, I = i32, B = Bool, C = Compiler> =
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
pub type Wgpu<F = f32, I = i32, B = Bool, C = Compiler> =
    JitBackend<cubecl::wgpu::WgpuRuntime<C>, F, I, B>;

#[cfg(test)]
mod tests {
    use burn_jit::JitBackend;
    #[cfg(feature = "spirv")]
    pub use half::f16;
    pub type TestRuntime = cubecl::wgpu::WgpuRuntime<super::Compiler>;

    // Don't test `flex32` for now, burn sees it as `f32` but is actually `f16` precision, so it
    // breaks a lot of tests from precision issues
    #[cfg(feature = "spirv")]
    burn_jit::testgen_all!([f16, f32], [i8, i16, i32, i64], [u8, u32]);
    #[cfg(not(feature = "spirv"))]
    burn_jit::testgen_all!([f32], [i32], [u32]);
}
