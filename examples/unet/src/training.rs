use crate::brain_tumor_data::BrainTumorBatcher;
use crate::brain_tumor_data::BrainTumorDataset;
use crate::unet_model::{UNet, UNetConfig};
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::AutodiffModule;
use burn::module::Module;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Config;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Float, Int, Tensor};
use std::path::Path;

#[derive(Config)]
pub struct UNetTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64, // TODO: explore using LrScheduler as is done in Learner? https://burn.dev/docs/burn/lr_scheduler/trait.LrScheduler.html
    pub model: UNetConfig,
    pub optimizer: AdamConfig,
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &Path) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &Path,
    config: UNetTrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    // Save training config
    config
        .save(artifact_dir.join("config.json")) // .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);

    // Create the model and optimizer
    let mut model: UNet<B> = config.model.init::<B>(device);
    let mut optim = config.optimizer.init::<B, UNet<B>>();

    // Create the batcher.
    let batcher_train = BrainTumorBatcher::<B>::new(device.clone());
    let batcher_valid = BrainTumorBatcher::<B::InnerBackend>::new(device.clone());
    let batcher_test = BrainTumorBatcher::<B::InnerBackend>::new(device.clone());

    // Create the datasets
    let train_dataset: BrainTumorDataset =
        BrainTumorDataset::train().expect("Failed to build training dataset");
    let valid_dataset: BrainTumorDataset =
        BrainTumorDataset::valid().expect("Failed to build validation dataset");
    let test_dataset: BrainTumorDataset =
        BrainTumorDataset::test().expect("Failed to build test dataset");

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let _dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let loss_config = BinaryCrossEntropyLossConfig::new().with_logits(true);
    // Iterate over our training and validation loop for X epochs.
    for epoch in 0..config.num_epochs {
        // Implement the training loop

        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output: Tensor<B, 4, Float> = model.forward(batch.source_tensor);
            // pixel-wise binary cross entropy loss
            let output_flat: Tensor<B, 2, Float> = output.flatten(1, 3);
            let target_flat: Tensor<B, 2, Int> = batch.target_tensor.flatten(1, 3);

            let loss: Tensor<B, 1> = loss_config
                .init(device)
                .forward(output_flat.clone(), target_flat.clone());

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff
        let model_valid = model.valid();

        // Implement the validation loop
        for (iteration, batch) in dataloader_valid.iter().enumerate() {
            let output = model_valid.forward(batch.source_tensor);
            // pixel-wise binary cross entropy loss
            let output_flat: Tensor<B::InnerBackend, 2, Float> = output.flatten(1, 3);
            let target_flat: Tensor<B::InnerBackend, 2, Int> = batch.target_tensor.flatten(1, 3);
            let loss: Tensor<B::InnerBackend, 1> = loss_config
                .init(device)
                .forward(output_flat.clone(), target_flat.clone());

            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );
        }
    }

    // Save the trained model
    model
        .save_file(artifact_dir.join("UNet"), &CompactRecorder::new())
        .expect("UNet saved.");
}
