use burn::autodiff::ADBackendDecorator;
use burn::wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

fn main() {
    custom_training_loop::run::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, f32, i32>>>(
        WgpuDevice::default(),
    );
}
