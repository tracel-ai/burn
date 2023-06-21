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
    fn backend() -> wgpu::Backend;
}

#[derive(Default, Debug, Clone)]
pub struct Vulkan;

#[derive(Default, Debug, Clone)]
pub struct Metal;

#[derive(Default, Debug, Clone)]
pub struct OpenGl;

#[derive(Default, Debug, Clone)]
pub struct Dx11;

#[derive(Default, Debug, Clone)]
pub struct Dx12;

#[derive(Default, Debug, Clone)]
pub struct WebGpu;

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
