use burn::backend::{Autodiff, WebGpu, wgpu::WgpuDevice};

fn main() {
    custom_renderer::run::<Autodiff<WebGpu>>(WgpuDevice::default());
}
