use burn::backend::wgpu::WgpuDevice;

fn main() {
    custom_learning_strategy::training::run(WgpuDevice::default());
}
