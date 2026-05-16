use burn::backend::wgpu::WgpuDevice;

fn main() {
    custom_training_loop::run(WgpuDevice::default());
}
