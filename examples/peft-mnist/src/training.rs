use crate::{
    data::{MnistBatch, MnistBatcher},
    model::{LoRAMLP, SimpleMLP},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

const ARTIFACT_DIR: &str = "./tmp/peft-mnist";

/// Training step implementation for LoRA model
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for LoRAMLP<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

/// Validation step implementation for LoRA model
impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for LoRAMLP<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 3)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 8)]
    pub lora_rank: usize,
    #[config(default = 16.0)]
    pub lora_alpha: f64,
    pub optimizer: AdamConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new(AdamConfig::new())
    }
}

pub fn train<B: AutodiffBackend>(device: B::Device) {
    // Create artifact directory
    std::fs::create_dir_all(ARTIFACT_DIR).ok();

    let config = TrainingConfig::default();
    B::seed(&device, config.seed);

    println!("🔥 Burn PEFT MNIST Example");
    println!("========================================");
    println!("This example demonstrates parameter-efficient fine-tuning");
    println!("using LoRA on the MNIST dataset.\n");

    // Load MNIST dataset
    println!("📦 Loading MNIST dataset...");
    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Create a baseline model (normally this would be pretrained)
    println!("📊 Creating baseline MLP...");
    let baseline = SimpleMLP::<B>::new(&device);

    // Count baseline parameters
    let baseline_params = baseline.num_params();
    println!("   Baseline parameters: {}", baseline_params);

    // Convert to LoRA for fine-tuning
    println!(
        "\n🎯 Converting to LoRA (rank={}, alpha={})...",
        config.lora_rank, config.lora_alpha
    );
    let lora_model =
        LoRAMLP::from_pretrained(baseline, config.lora_rank, config.lora_alpha, &device);

    // Count trainable parameters (in LoRA, only adapter matrices are trainable)
    let lora_params = lora_model.num_params();
    let trainable_ratio = 1.0 - (lora_params as f64 / baseline_params as f64);
    println!("   LoRA total parameters: {}", lora_params);
    println!("   Parameter reduction: {:.1}%", trainable_ratio * 100.0);

    println!(
        "\n🚀 Training LoRA model for {} epochs...",
        config.num_epochs
    );

    // Setup learner
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            lora_model,
            config.optimizer.init(),
            config.learning_rate,
            LearningStrategy::SingleDevice(device),
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    // Save the trained model
    result
        .model
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    // Save configuration
    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .expect("Failed to save config");

    println!("\n✅ Training complete!");
    println!("   Model saved to: {}/model", ARTIFACT_DIR);
    println!("   Config saved to: {}/config.json", ARTIFACT_DIR);
    println!("\n   Run inference: cargo run --release -- infer");
}
