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
type WgpuInner<C, F, I, B> =
    burn_fusion::Fusion<CubeBackend<cubecl::wgpu::WgpuRuntime<C>, F, I, B>>;

#[cfg(not(feature = "fusion"))]
type WgpuInner<C, F, I, B> = CubeBackend<cubecl::wgpu::WgpuRuntime<C>, F, I, B>;

pub type Wgpu<F = f32, I = i32, B = u32> = WgpuInner<AutoCompiler, F, I, B>;

#[cfg(feature = "vulkan")]
/// Tensor backend that leverages the Vulkan graphics API to execute GPU compute shaders compiled to SPIR-V.
pub type Vulkan<F = f32, I = i32, B = u8> = WgpuInner<SpirvCompiler, F, I, B>;

#[cfg(feature = "webgpu")]
/// Tensor backend that uses the wgpu crate to execute GPU compute shaders written in WGSL.
pub type WebGpu<F = f32, I = i32, B = u32> = WgpuInner<WgslCompiler, F, I, B>;

#[cfg(feature = "metal")]
/// Tensor backend that leverages the Metal graphics API to execute GPU compute shaders compiled to MSL.
pub type Metal<F = f32, I = i32, B = u8> = WgpuInner<MslCompiler, F, I, B>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, QTensorPrimitive};

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
        assert!(!B::supports_dtype(&device, DType::Bool(BoolStore::Native)));

        #[cfg(feature = "vulkan")]
        {
            assert!(B::supports_dtype(&device, DType::F16));
            assert!(B::supports_dtype(&device, DType::I16));
            assert!(B::supports_dtype(&device, DType::I8));
            assert!(B::supports_dtype(&device, DType::U16));
            assert!(B::supports_dtype(&device, DType::U8));

            assert!(!B::supports_dtype(&device, DType::F64));
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
