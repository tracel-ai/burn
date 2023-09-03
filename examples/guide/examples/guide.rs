use burn::optim::AdamConfig;
use burn_dataset::Dataset;
use guide::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = burn::wgpu::WgpuBackend<burn::wgpu::AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = burn::autodiff::ADBackendDecorator<MyBackend>;

    let device = burn::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    guide::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    guide::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn_dataset::source::huggingface::MNISTDataset::test()
            .get(42)
            .unwrap(),
    );
}
