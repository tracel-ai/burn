use burn::backend::{wgpu::WgpuDevice, Autodiff, WebGpu};

fn main() {
    custom_renderer::run::<Autodiff<WebGpu>>(WgpuDevice::default());
}
