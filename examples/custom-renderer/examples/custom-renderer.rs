use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};

fn main() {
    custom_renderer::run::<Autodiff<Wgpu>>(WgpuDevice::default());
}
