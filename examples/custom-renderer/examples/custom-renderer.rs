use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};

fn main() {
    custom_renderer::run::<Autodiff<Wgpu>>(WgpuDevice::default());
}
