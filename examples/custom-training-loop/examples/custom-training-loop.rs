use burn::backend::{wgpu::WgpuDevice, Autodiff, HttpBackend, Wgpu};

fn main() {
    custom_training_loop::run::<Autodiff<HttpBackend>>(Default::default());
}
