use burn_autodiff::ADBackendDecorator;
use burn_wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

fn main() {
    custom_training_loop::run::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, f32, i32>>>(
        WgpuDevice::default(),
    );
}
