#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

#[cfg(feature = "template")]
pub use burn_cubecl::{
    kernel::{KernelMetadata, into_contiguous},
    kernel_source,
    template::{KernelSource, SourceKernel, SourceTemplate, build_info},
};

pub use burn_cubecl::{BoolElement, FloatElement, IntElement};
pub use burn_cubecl::{CubeBackend, tensor::CubeTensor};
pub use cubecl::CubeDim;
pub use cubecl::flex32;

#[cfg(feature = "metal")]
use cubecl::wgpu::MslCompiler;
#[cfg(feature = "vulkan")]
use cubecl::wgpu::SpirvCompiler;
#[cfg(feature = "webgpu")]
use cubecl::wgpu::WgslCompiler;

pub use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuResource, WgpuRuntime,
    WgpuSetup, WgpuStorage, init_device, init_setup, init_setup_async,
};
// Vulkan and WebGpu would have conflicting type names
pub mod graphics {
    pub use cubecl::wgpu::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan, WebGpu};
}

#[cfg(feature = "fusion")]
type WgpuInner<C> = burn_fusion::Fusion<CubeBackend<cubecl::wgpu::WgpuRuntime<C>>>;

#[cfg(not(feature = "fusion"))]
type WgpuInner<C> = CubeBackend<cubecl::wgpu::WgpuRuntime<C>>;

/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan][crate::graphics::Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::graphics::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::graphics::Dx12) on Windows.
///   - [Metal][crate::graphics::Metal] on Apple hardware.
///   - [WebGPU](crate::graphics::WebGpu) on supported browsers and `wasm` runtimes.
///
/// The selected graphics API is chosen automatically at runtime, and the appropriate shader
/// compiler (WGSL, SPIR-V or MSL) is dispatched via [`AutoCompiler`]. When the target API is
/// known ahead of time, prefer the dedicated `Vulkan`, `WebGpu` or `Metal` backend aliases
/// (enabled by their respective Cargo features), which lock the compiler at compile time and
/// avoid the runtime dispatch.
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
/// When the `fusion` feature flag is enabled (the default), this backend uses [burn_fusion] to
/// compile and optimize streams of tensor operations for improved performance. You can disable
/// the `fusion` feature flag to remove that functionality, which might be necessary on `wasm`
/// for now.
pub type Wgpu = WgpuInner<AutoCompiler>;

/// Tensor backend that leverages the Vulkan graphics API to execute GPU compute shaders compiled to SPIR-V.
///
/// This is a specialization of [`Wgpu`] that pins the shader compiler to SPIR-V at compile time,
/// removing the runtime [`AutoCompiler`] dispatch. Enable the `vulkan` feature to use it.
/// Multiple wgpu backend aliases (`Vulkan`, `WebGpu`, `Metal`) can be enabled simultaneously
/// since each is a distinct type parameterized by its own compiler.
#[cfg(feature = "vulkan")]
pub type Vulkan = WgpuInner<SpirvCompiler>;

/// Tensor backend that uses the wgpu crate to execute GPU compute shaders written in WGSL.
///
/// This is a specialization of [`Wgpu`] that pins the shader compiler to WGSL at compile time,
/// removing the runtime [`AutoCompiler`] dispatch. Enable the `webgpu` feature to use it.
#[cfg(feature = "webgpu")]
pub type WebGpu = WgpuInner<WgslCompiler>;

/// Tensor backend that leverages the Metal graphics API to execute GPU compute shaders compiled to MSL.
///
/// This is a specialization of [`Wgpu`] that pins the shader compiler to MSL at compile time,
/// removing the runtime [`AutoCompiler`] dispatch. Enable the `metal` feature to use it.
#[cfg(feature = "metal")]
pub type Metal = WgpuInner<MslCompiler>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, DeviceOps};

    #[test]
    fn should_support_dtypes() {
        type B = Wgpu;
        let device = WgpuDevice::default();
        let scheme = device.defaults().quantization.scheme;

        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::QFloat(scheme)));
        assert!(!B::supports_dtype(&device, DType::Bool(BoolStore::Native)));

        #[cfg(feature = "vulkan")]
        {
            assert!(B::supports_dtype(&device, DType::F16));
            assert!(B::supports_dtype(&device, DType::I16));
            assert!(B::supports_dtype(&device, DType::I8));
            assert!(B::supports_dtype(&device, DType::U16));
            assert!(B::supports_dtype(&device, DType::U8));

            // NOTE: F64 is not part of the default types, but is supported based on `shader_float64` feature
            assert!(B::supports_dtype(&device, DType::F64));
            assert!(!B::supports_dtype(&device, DType::Flex32));
            // Not supported for any arithmetics, but buffer, conversion and possibly matmul (hw dependent)
            assert!(!B::supports_dtype(&device, DType::BF16));
        }

        #[cfg(feature = "metal")]
        {
            assert!(B::supports_dtype(&device, DType::F16));
            assert!(B::supports_dtype(&device, DType::I16));
            assert!(B::supports_dtype(&device, DType::I8));
            assert!(B::supports_dtype(&device, DType::U16));
            assert!(B::supports_dtype(&device, DType::U8));

            assert!(!B::supports_dtype(&device, DType::F64));
            assert!(!B::supports_dtype(&device, DType::BF16));
            assert!(!B::supports_dtype(&device, DType::Flex32));
        }

        // On macOS without the `metal` feature, wgpu still uses Metal at runtime,
        // which doesn't support F64 or BF16.
        #[cfg(all(not(any(feature = "vulkan", feature = "metal")), target_os = "macos"))]
        {
            assert!(B::supports_dtype(&device, DType::Flex32));
            assert!(B::supports_dtype(&device, DType::F16));

            assert!(!B::supports_dtype(&device, DType::F64));
            assert!(!B::supports_dtype(&device, DType::BF16));
            assert!(!B::supports_dtype(&device, DType::I16));
            assert!(!B::supports_dtype(&device, DType::I8));
            assert!(!B::supports_dtype(&device, DType::U16));
            assert!(!B::supports_dtype(&device, DType::U8));
        }

        #[cfg(not(any(feature = "vulkan", feature = "metal", target_os = "macos")))]
        {
            assert!(B::supports_dtype(&device, DType::F64));
            assert!(B::supports_dtype(&device, DType::Flex32));
            assert!(B::supports_dtype(&device, DType::F16));

            assert!(!B::supports_dtype(&device, DType::BF16));
            assert!(!B::supports_dtype(&device, DType::I16));
            assert!(!B::supports_dtype(&device, DType::I8));
            assert!(!B::supports_dtype(&device, DType::U16));
            assert!(!B::supports_dtype(&device, DType::U8));
        }
    }
}
