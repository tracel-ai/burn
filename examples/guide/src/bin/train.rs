use burn::{
    backend::{Autodiff, WebGpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};
use guide::{
    inference,
    model::ModelConfig,
    training::{self, TrainingConfig},
};

fn main() {
    type MyBackend = WebGpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/guide";

    // Train the model
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    // Infer the model
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
