use burn::backend::wgpu::WgpuDevice;
use burn::backend::WgpuAutodiffBackend;

fn main() {
    custom_renderer::run::<WgpuAutodiffBackend>(WgpuDevice::default());
}
