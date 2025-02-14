use crate::brain_tumor_data::BrainTumorBatch;
use crate::brain_tumor_data::BrainTumorBatcher;
use crate::brain_tumor_data::BrainTumorDataset;
use crate::unet_model::{UNet, UNetConfig};
use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use burn::prelude::Config;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Float;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::train::metric::HammingScore;
use burn::train::metric::LossMetric;
use burn::train::Learner;
use burn::train::LearnerBuilder;
use burn::train::MultiLabelClassificationOutput;
use burn::train::TrainOutput;
use burn::train::TrainStep;
use burn::train::ValidStep;
use std::path::Path;

impl<B: Backend> UNet<B> {
    pub fn forward_segmentation(
        &self,
        source_tensor: Tensor<B, 4>,
        target_tensor: Tensor<B, 4, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output: Tensor<B, 4, Float> = self.forward(source_tensor);
        // pixel-wise binary cross entropy loss
        let output_flat: Tensor<B, 2, Float> = output.flatten(1, 3);
        let target_flat: Tensor<B, 2, Int> = target_tensor.flatten(1, 3);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output_flat.device())
            .forward(output_flat.clone(), target_flat.clone());

        MultiLabelClassificationOutput::new(loss, output_flat, target_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<BrainTumorBatch<B>, MultiLabelClassificationOutput<B>>
    for UNet<B>
{
    fn step(&self, batch: BrainTumorBatch<B>) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_segmentation(batch.source_tensor, batch.target_tensor);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BrainTumorBatch<B>, MultiLabelClassificationOutput<B>> for UNet<B> {
    fn step(&self, batch: BrainTumorBatch<B>) -> MultiLabelClassificationOutput<B> {
        self.forward_segmentation(batch.source_tensor, batch.target_tensor)
    }
}

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
    pub learning_rate: f64,
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
        .save(artifact_dir.join("config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);

    // Create the batcher.
    let batcher_train = BrainTumorBatcher::<B>::new(device.clone());
    let batcher_valid = BrainTumorBatcher::<B::InnerBackend>::new(device.clone());

    // Create the datasets
    let train_dataset: BrainTumorDataset =
        BrainTumorDataset::train().expect("Failed to build training dataset");
    let valid_dataset: BrainTumorDataset =
        BrainTumorDataset::valid().expect("Failed to build validation dataset");

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

    let learner: Learner<_> = LearnerBuilder::new(artifact_dir.to_path_buf())
        .metric_train_numeric(HammingScore::new())
        .metric_valid_numeric(HammingScore::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),       // Initialize the model
            config.optimizer.init::<B, UNet<B>>(), // Initialize the optimizer
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save the trained model
    model_trained
        .save_file(artifact_dir.join("UNet"), &CompactRecorder::new())
        .expect("Trained UNet model should be saved successfully");
}
