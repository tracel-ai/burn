use burn::backend::wgpu::WgpuDevice;

fn main() {
    custom_renderer::run(WgpuDevice::default());
}
