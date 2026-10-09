#![cfg_attr(docsrs, feature(doc_cfg))]

//! The [wgpu](https://github.com/gfx-rs/wgpu) runtime for [Burn](https://github.com/tracel-ai/burn)'s
//! CubeCL backend: Vulkan, Metal, DirectX 12, OpenGL and WebGPU through one API.
//!
//! Applications enable Burn's `wgpu` feature and create a device with `Device::wgpu(..)`, which
//! picks a graphics API for the platform, or configure it first with `Device::wgpu_options()`.
//! The `vulkan`, `metal` and `webgpu` features add devices pinned to that API. Applications do not
//! need this crate directly.
//!
//! This crate also re-exports the wgpu runtime types ([`WgpuDevice`], [`WgpuSetup`],
//! [`MemoryConfiguration`], ...) for sharing an existing wgpu device with Burn, and, with the
//! `template` feature, the API for launching hand-written WGSL kernels.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `fusion` (default): kernel fusion.
//! - `autotune` (default): benchmark kernel variants at runtime and keep the fastest.
//! - `vulkan`, `metal`, `webgpu`: devices pinned to that graphics API, using its native shader
//!   compiler where available.
//! - `template`: launch hand-written kernels.
//! - `exclusive-memory-only`: never share a memory page between allocations (always the case on
//!   wasm).
//! - `tracing`: instrument operations with the `tracing` crate.

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
pub use cubecl::wgpu::MslCompiler;
#[cfg(not(target_family = "wasm"))]
pub use cubecl::wgpu::try_init_setup;
pub use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuBackend, WgpuDevice, WgpuInitError,
    WgpuResource, WgpuRuntime, WgpuSetup, WgpuStorage, init_device, init_device_with_api,
    init_setup, init_setup_async, try_init_device, try_init_device_with_api, try_init_setup_async,
    wgpu,
};
// Vulkan and WebGpu would have conflicting type names
pub mod graphics {
    pub use cubecl::wgpu::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan, WebGpu};
}

#[cfg(feature = "fusion")]
type WgpuInner = burn_fusion::Fusion<CubeBackend>;

#[cfg(not(feature = "fusion"))]
type WgpuInner = CubeBackend;

/// Tensor backend that uses the wgpu crate for executing GPU compute shaders.
///
/// This backend can target multiple graphics APIs, including:
///   - [Vulkan][crate::graphics::Vulkan] on Linux, Windows, and Android.
///   - [OpenGL](crate::graphics::OpenGl) on Linux, Windows, and Android.
///   - [DirectX 12](crate::graphics::Dx12) on Windows.
///   - [Metal][crate::graphics::Metal] on Apple hardware.
///   - [WebGPU](crate::graphics::WebGpu) on supported browsers and `wasm` runtimes.
///
/// Automatic devices use [`AutoCompiler`] to select WGSL, SPIR-V or MSL, with WGSL fallback
/// when a native compiler is unavailable. Enabling `metal` makes MSL available without changing
/// the compiler used by unrelated devices.
///
/// With `metal` enabled, an explicitly selected Metal device, created with Burn's
/// `burn::tensor::Device::metal` or [`cubecl::Device::metal_msl`], requires native MSL and panics
/// during initialization if it is unavailable. The same applies to [`init_setup`] or
/// [`init_setup_async`] with [`graphics::Metal`], and to [`init_device_with_api`] with
/// [`graphics::Metal`]. Importing through [`init_device`] retains automatic compiler selection
/// and fallback. The corresponding `try_*` functions return initialization errors instead
/// of panicking.
///
/// Multiple backend features can be enabled together. The deprecated `Vulkan`, `WebGpu` and
/// `Metal` aliases name the same backend; use [`Wgpu`] with an explicit device constructor to
/// select the graphics API. Compiler selection follows the device and enabled features.
///
/// Application code configures and initializes the runtime with `Device::wgpu_options`:
///
/// ```rust, ignore
/// use burn::tensor::{Device, wgpu::WgpuBackend};
/// let device = Device::wgpu_options()
///     .graphics_api(WgpuBackend::Vulkan)
///     .tasks_max(32)
///     .init()?;
/// ```
/// Use `init_async().await` for native async applications and browsers, or
/// `.setup(existing_setup).init()` to share an application's existing device and queue.
/// Initialize once, then clone the Burn device to share its runtime. The low-level
/// [`try_init_setup_async`] and [`try_init_device`] APIs remain available for backend authors.
///
/// # Notes
///
/// When the `fusion` feature flag is enabled (the default), this backend uses [burn_fusion] to
/// compile and optimize streams of tensor operations for improved performance. You can disable
/// the `fusion` feature flag to remove that functionality, which might be necessary on `wasm`
/// for now.
pub type Wgpu = WgpuInner;

/// Deprecated alias of [`Wgpu`].
///
/// Select Vulkan explicitly with Burn's `burn::tensor::Device::vulkan` or
/// [`cubecl::Device::vulkan`]. Compiler selection follows the device; see [`Wgpu`].
#[cfg(feature = "vulkan")]
#[deprecated(
    since = "0.22.0",
    note = "Use `Wgpu` with `burn::tensor::Device::vulkan` to select Vulkan explicitly. This alias is identical to `Wgpu` and does not select a graphics API."
)]
pub type Vulkan = WgpuInner;

/// Deprecated alias of [`Wgpu`].
///
/// Select WebGPU explicitly with Burn's `burn::tensor::Device::webgpu` or
/// [`cubecl::Device::webgpu`]. Compiler selection follows the device; see [`Wgpu`].
#[cfg(feature = "webgpu")]
#[deprecated(
    since = "0.22.0",
    note = "Use `Wgpu` with `burn::tensor::Device::webgpu` to select WebGPU explicitly. This alias is identical to `Wgpu` and does not select a graphics API."
)]
pub type WebGpu = WgpuInner;

/// Deprecated alias of [`Wgpu`].
///
/// Use Burn's `burn::tensor::Device::metal` or
/// [`cubecl::Device::metal_msl`] to select Metal explicitly. Both select
/// [`WgpuBackend::Metal`], which requires native MSL support with `metal` enabled and panics
/// during initialization if it is unavailable. Automatic devices retain WGSL fallback.
///
/// To import an existing Metal setup with the same requirement, use
/// `init_device_with_api::<graphics::Metal>(setup, RuntimeOptions::default())`.
#[cfg(feature = "metal")]
#[deprecated(
    since = "0.22.0",
    note = "Use `Wgpu` with `burn::tensor::Device::metal` to select Metal explicitly. This alias is identical to `Wgpu` and does not select a graphics API."
)]
pub type Metal = WgpuInner;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, DeviceOps};

    fn assert_common_dtypes(device: &cubecl::Device) {
        // Metal and Vulkan alias Wgpu; the device selects the compiler.
        type B = Wgpu;
        let defaults = device.defaults();
        let scheme = defaults.quantization.scheme;

        assert!(B::supports_dtype(device, DType::F32));
        assert!(B::supports_dtype(device, DType::F16));
        assert!(B::supports_dtype(device, DType::I64));
        assert!(B::supports_dtype(device, DType::I32));
        assert!(B::supports_dtype(device, DType::U64));
        assert!(B::supports_dtype(device, DType::U32));
        assert!(B::supports_dtype(device, DType::QFloat(scheme)));
        assert!(!B::supports_dtype(device, DType::Bool(BoolStore::Native)));
        assert!(B::supports_dtype(device, defaults.bool_dtype.into()));
    }

    #[cfg(any(
        all(feature = "vulkan", not(target_family = "wasm")),
        all(feature = "metal", target_vendor = "apple")
    ))]
    fn assert_fp4_dtypes(device: &cubecl::Device) {
        use burn_backend::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

        // FP8 block scale storage and software conversion support NVFP4 and MXFP4.
        let fp4 = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .with_store(QuantStore::PackedU32(0));
        let nvfp4 = fp4
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32);
        let mxfp4 = fp4.per_block([32], ScaleDtype::UE8M0);
        assert!(Wgpu::supports_dtype(device, DType::QFloat(nvfp4)));
        assert!(Wgpu::supports_dtype(device, DType::QFloat(mxfp4)));
    }

    #[test]
    fn should_support_dtypes() {
        let device = cubecl::Device::Wgpu(WgpuDevice::default());
        assert_common_dtypes(&device);
    }

    #[cfg(all(feature = "vulkan", not(target_family = "wasm")))]
    #[test]
    fn should_support_vulkan_dtypes() {
        type B = Wgpu;
        let device = cubecl::Device::Wgpu(WgpuDevice::default().on(WgpuBackend::Vulkan));
        assert_common_dtypes(&device);

        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));

        // F64 is supported through the shader_float64 feature.
        assert!(B::supports_dtype(&device, DType::F64));
        assert!(!B::supports_dtype(&device, DType::Flex32));
        // BF16 supports storage and conversion, but not all scalar arithmetic operations.
        assert!(!B::supports_dtype(&device, DType::BF16));

        assert_fp4_dtypes(&device);
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn should_support_metal_dtypes() {
        type B = Wgpu;
        let device = cubecl::Device::Wgpu(WgpuDevice::default().on(WgpuBackend::Metal));
        assert_common_dtypes(&device);

        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));

        assert!(!B::supports_dtype(&device, DType::F64));
        // MSL 3.2 carries bfloat natively.
        assert!(B::supports_dtype(&device, DType::BF16));
        assert!(!B::supports_dtype(&device, DType::Flex32));

        assert_fp4_dtypes(&device);
    }

    #[cfg(not(any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn should_support_wgsl_dtypes() {
        type B = Wgpu;
        let device = cubecl::Device::Wgpu(WgpuDevice::default());
        assert_common_dtypes(&device);

        assert!(B::supports_dtype(&device, DType::Flex32));
        #[cfg(target_os = "macos")]
        assert!(!B::supports_dtype(&device, DType::F64));
        #[cfg(not(target_os = "macos"))]
        assert!(B::supports_dtype(&device, DType::F64));
        assert!(!B::supports_dtype(&device, DType::BF16));
        assert!(!B::supports_dtype(&device, DType::I16));
        assert!(!B::supports_dtype(&device, DType::I8));
        assert!(!B::supports_dtype(&device, DType::U16));
        assert!(!B::supports_dtype(&device, DType::U8));
    }
}
