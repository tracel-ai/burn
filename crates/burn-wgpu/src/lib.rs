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

pub use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuResource, WgpuRuntime,
    WgpuSetup, WgpuStorage, init_device, init_setup, init_setup_async,
};
// Vulkan and WebGpu would have conflicting type names
pub mod graphics {
    pub use cubecl::wgpu::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan, WebGpu};
}

#[cfg(feature = "cubecl-wgsl")]
pub use cubecl::wgpu::WgslCompiler;
#[cfg(feature = "cubecl-spirv")]
pub use cubecl::wgpu::vulkan::VkSpirvCompiler;

#[cfg(feature = "fusion")]
/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan][crate::graphics::Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::graphics::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::graphics::Dx12) on Windows.
///   - [Metal][crate::graphics::Metal] on Apple hardware.
///   - [WebGPU](crate::graphics::WebGpu) on supported browsers and `wasm` runtimes.
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
pub type Wgpu<F = f32, I = i32, B = u32> =
    burn_fusion::Fusion<CubeBackend<cubecl::wgpu::WgpuRuntime, F, I, B>>;

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
pub type Wgpu<F = f32, I = i32, B = u32> = CubeBackend<cubecl::wgpu::WgpuRuntime, F, I, B>;

#[cfg(feature = "vulkan")]
/// Tensor backend that leverages the Vulkan graphics API to execute GPU compute shaders compiled to SPIR-V.
pub type Vulkan<F = f32, I = i32, B = u8> = Wgpu<F, I, B>;

#[cfg(feature = "webgpu")]
/// Tensor backend that uses the wgpu crate to execute GPU compute shaders written in WGSL.
pub type WebGpu<F = f32, I = i32, B = u32> = Wgpu<F, I, B>;

#[cfg(feature = "metal")]
/// Tensor backend that leverages the Metal graphics API to execute GPU compute shaders compiled to MSL.
pub type Metal<F = f32, I = i32, B = u8> = Wgpu<F, I, B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, DType, QTensorPrimitive};

    #[test]
    fn should_support_dtypes() {
        type B = Wgpu;
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(CubeTensor::<WgpuRuntime>::default_scheme())
        ));
        // Registered as supported type but we don't actually use it?
        assert!(B::supports_dtype(&device, DType::Bool));

        #[cfg(feature = "vulkan")]
        {
            assert!(B::supports_dtype(&device, DType::F16));
            assert!(B::supports_dtype(&device, DType::BF16));
            assert!(B::supports_dtype(&device, DType::I16));
            assert!(B::supports_dtype(&device, DType::I8));
            assert!(B::supports_dtype(&device, DType::U16));
            assert!(B::supports_dtype(&device, DType::U8));

            assert!(!B::supports_dtype(&device, DType::F64));
            assert!(!B::supports_dtype(&device, DType::Flex32));
        }

        #[cfg(feature = "metal")]
        {
            assert!(B::supports_dtype(&device, DType::F16));
            assert!(B::supports_dtype(&device, DType::I16));
            assert!(B::supports_dtype(&device, DType::U16));

            assert!(!B::supports_dtype(&device, DType::F64));
            assert!(!B::supports_dtype(&device, DType::Flex32));
            assert!(!B::supports_dtype(&device, DType::I8));
            assert!(!B::supports_dtype(&device, DType::U8));
        }

        #[cfg(not(any(feature = "vulkan", feature = "metal")))]
        {
            assert!(B::supports_dtype(&device, DType::F64));
            assert!(B::supports_dtype(&device, DType::Flex32));

            assert!(!B::supports_dtype(&device, DType::F16));
            assert!(!B::supports_dtype(&device, DType::BF16));
            assert!(!B::supports_dtype(&device, DType::I16));
            assert!(!B::supports_dtype(&device, DType::I8));
            assert!(!B::supports_dtype(&device, DType::U16));
            assert!(!B::supports_dtype(&device, DType::U8));
        }
    }
}
