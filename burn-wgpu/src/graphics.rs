/// The basic trait to specify which graphics API to use as Backend.
///
/// Options are:
///   - [Vulkan](Vulkan)
///   - [Metal](Metal)
///   - [OpenGL](OpenGl)
///   - [DirectX 11](Dx11)
///   - [DirectX 12](Dx12)
///   - [WebGpu](WebGpu)
pub trait GraphicsApi: Send + Sync + core::fmt::Debug + Default + Clone + 'static {
    /// The wgpu backend.
    fn backend() -> wgpu::Backend;
}

/// Vulkan graphics API.
#[derive(Default, Debug, Clone)]
pub struct Vulkan;

/// Metal graphics API.
#[derive(Default, Debug, Clone)]
pub struct Metal;

/// OpenGL graphics API.
#[derive(Default, Debug, Clone)]
pub struct OpenGl;

/// DirectX 11 graphics API.
#[derive(Default, Debug, Clone)]
pub struct Dx11;

/// DirectX 12 graphics API.
#[derive(Default, Debug, Clone)]
pub struct Dx12;

/// WebGpu graphics API.
#[derive(Default, Debug, Clone)]
pub struct WebGpu;

/// Automatic graphics API based on OS.
#[derive(Default, Debug, Clone)]
pub struct AutoGraphicsApi;

impl GraphicsApi for Vulkan {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Vulkan
    }
}

impl GraphicsApi for Metal {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Metal
    }
}

impl GraphicsApi for OpenGl {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Gl
    }
}

impl GraphicsApi for Dx11 {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Dx11
    }
}

impl GraphicsApi for Dx12 {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Dx12
    }
}

impl GraphicsApi for WebGpu {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::BrowserWebGpu
    }
}

impl GraphicsApi for AutoGraphicsApi {
    fn backend() -> wgpu::Backend {
        #[cfg(target_os = "macos")]
        return wgpu::Backend::Metal;
        #[cfg(not(target_os = "macos"))]
        wgpu::Backend::Vulkan
    }
}
