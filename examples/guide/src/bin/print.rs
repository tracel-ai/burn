use guide::model::ModelConfig;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let model = ModelConfig::new(10, 512).init(&device.into());

    println!("{model}");
}
