use std::path::Path;

use burn::backend::wgpu::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use unet::training::UNetTrainingConfig;
use unet::unet_model::UNetConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    // Model training
    let training_artifact_dir = Path::new("artifacts");
    let model_config = UNetConfig::new();
    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);
    let training_config = UNetTrainingConfig::new(model_config, optimizer_config)
        .with_num_epochs(1)
        .with_batch_size(4)
        .with_num_workers(1)
        .with_seed(42)
        .with_learning_rate(1e-3);

    unet::training::train::<Autodiff<MyBackend>>(training_artifact_dir, training_config, &device);
}
