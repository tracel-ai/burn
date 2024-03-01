use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};

fn main() {
    custom_training_loop::run::<Autodiff<Wgpu>>(WgpuDevice::default());
}
