use burn::prelude::*;
use guide::model::ModelConfig;

fn main() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let model = ModelConfig::new(10, 512).init(&device);

    println!("{model}");
}
