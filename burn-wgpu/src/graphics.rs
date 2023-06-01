/// The basic trait to specify which graphics API to use as Backend.
///
/// Options are:
///   - [Vulkan](Vulkan)
///   - [Metal](Metal)
///   - [OpenGL](OpenGL)
///   - [DirectX 11](Dx11)
///   - [DirectX 12](Dx12)
///   - [WebGPU](WebGPU)
pub trait GraphicsAPI: Send + Sync + core::fmt::Debug + Default + Clone + 'static {
    fn backend() -> wgpu::Backend;
}

#[derive(Default, Debug, Clone)]
pub struct Vulkan;
#[derive(Default, Debug, Clone)]
pub struct Metal;
#[derive(Default, Debug, Clone)]
pub struct OpenGL;
#[derive(Default, Debug, Clone)]
pub struct Dx11;
#[derive(Default, Debug, Clone)]
pub struct Dx12;
#[derive(Default, Debug, Clone)]
pub struct WebGPU;

impl GraphicsAPI for Vulkan {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Vulkan
    }
}

impl GraphicsAPI for Metal {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Metal
    }
}

impl GraphicsAPI for OpenGL {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Gl
    }
}

impl GraphicsAPI for Dx11 {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Dx11
    }
}

impl GraphicsAPI for Dx12 {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::Dx12
    }
}

impl GraphicsAPI for WebGPU {
    fn backend() -> wgpu::Backend {
        wgpu::Backend::BrowserWebGpu
    }
}
