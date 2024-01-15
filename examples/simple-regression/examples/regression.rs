use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::WgpuDevice;
use regression::training::run;

fn main() {
    let device = WgpuDevice::default();
    run::<Autodiff<Wgpu>>(device);
}