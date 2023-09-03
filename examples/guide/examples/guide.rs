use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{WgpuAutodiffBackend, WgpuBackend};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use guide::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = WgpuAutodiffBackend<AutoGraphicsApi, f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    guide::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    guide::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::source::huggingface::MNISTDataset::test()
            .get(42)
            .unwrap(),
    );
}
