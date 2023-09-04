use burn::backend::wgpu::WgpuDevice;
use burn::backend::WgpuAutodiffBackend;

fn main() {
    custom_training_loop::run::<WgpuAutodiffBackend>(WgpuDevice::default());
}
