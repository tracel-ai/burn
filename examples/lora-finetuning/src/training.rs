//! Training configuration and execution for LoRA fine-tuning.
//!
//! This module contains:
//! - `LoraTrainingConfig`: Configurable training parameters
//! - `run()`: Main entry point for training
//! - TrainStep/InferenceStep implementations for the model

use crate::data::{SyntheticBatch, SyntheticBatcher, SyntheticDataset};
use crate::model::{
    SimpleMlp, SimpleMlpConfig, SimpleMlpWithLora, apply_lora, count_lora_trainable_params,
    count_params,
};

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::nn::lora::{LoraBias, LoraConfig};
use burn::nn::loss::MseLoss;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};

/// Artifact directory for saving training outputs.
static ARTIFACT_DIR: &str = "/tmp/burn-lora-example";

/// Configuration for LoRA fine-tuning training.
#[derive(Config, Debug)]
pub struct LoraTrainingConfig {
    // Model dimensions
    /// Input dimension for the MLP.
    #[config(default = 32)]
    pub d_input: usize,
    /// Hidden dimension for the MLP.
    #[config(default = 64)]
    pub d_hidden: usize,
    /// Output dimension for the MLP.
    #[config(default = 1)]
    pub d_output: usize,

    // LoRA parameters
    /// LoRA rank (low-rank dimension).
    #[config(default = 4)]
    pub lora_rank: usize,
    /// LoRA alpha (scaling factor).
    #[config(default = 8.0)]
    pub lora_alpha: f64,

    // Training parameters
    /// Number of training epochs.
    #[config(default = 100)]
    pub num_epochs: usize,
    /// Batch size for training.
    #[config(default = 32)]
    pub batch_size: usize,
    /// Learning rate.
    #[config(default = 1e-3)]
    pub learning_rate: f64,

    // Dataset parameters
    /// Number of training samples.
    #[config(default = 1000)]
    pub train_size: usize,
    /// Number of validation samples.
    #[config(default = 200)]
    pub valid_size: usize,
    /// Random seed for reproducibility.
    #[config(default = 42)]
    pub seed: u64,
}

impl<B: AutodiffBackend> TrainStep for SimpleMlpWithLora<B> {
    type Input = SyntheticBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: SyntheticBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward(batch.inputs);
        let loss = MseLoss::new().forward(
            output.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput::new(loss, output, batch.targets),
        )
    }
}

impl<B: Backend> InferenceStep for SimpleMlpWithLora<B> {
    type Input = SyntheticBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: SyntheticBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.inputs);
        let loss = MseLoss::new().forward(
            output.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );
        RegressionOutput::new(loss, output, batch.targets)
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Run LoRA fine-tuning with default configuration.
pub fn run<B: AutodiffBackend>(device: B::Device) {
    let config = LoraTrainingConfig::new();
    run_with_config::<B>(ARTIFACT_DIR, config, device);
}

/// Run LoRA fine-tuning with custom configuration.
pub fn run_with_config<B: AutodiffBackend>(
    artifact_dir: &str,
    config: LoraTrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);

    // Save config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    println!("=== LoRA Fine-tuning Example ===\n");

    // Create the base model (simulating a pre-trained model)
    let base_model =
        SimpleMlpConfig::new(config.d_input, config.d_hidden, config.d_output).init::<B>(&device);

    // Count parameters in base model
    let base_params = count_params(&base_model);
    println!("Base model parameters: {}", base_params);

    // Evaluate base model before fine-tuning using validation dataset
    let eval_dataset = SyntheticDataset::new(100, config.d_input);
    let (test_inputs, test_targets): (Vec<_>, Vec<_>) = (0..eval_dataset.len())
        .filter_map(|i| eval_dataset.get(i))
        .map(|item| {
            let input = Tensor::<B, 1>::from_floats(item.input.as_slice(), &device);
            let target = Tensor::<B, 1>::from_floats([item.target], &device);
            (input, target)
        })
        .unzip();

    let test_input = Tensor::stack(test_inputs, 0);
    let test_target = Tensor::stack(test_targets, 0);

    let base_output = base_model.forward(test_input.clone());
    let base_loss: f32 = MseLoss::new()
        .forward(
            base_output,
            test_target.clone(),
            burn::nn::loss::Reduction::Mean,
        )
        .into_scalar()
        .elem();
    println!("Base model loss (before fine-tuning): {:.6}", base_loss);

    // Save base model weights to disk (represents a pre-trained model checkpoint)
    let base_model_path = format!("{artifact_dir}/base_model");
    println!("\nSaving base model to: {}", base_model_path);
    base_model
        .clone()
        .save_file(&base_model_path, &CompactRecorder::new())
        .expect("Failed to save base model");

    // Configure LoRA
    let lora_config = LoraConfig::new(config.lora_rank)
        .with_alpha(config.lora_alpha)
        .with_dropout(0.0)
        .with_bias(LoraBias::None);

    println!("\nLoRA Configuration:");
    println!("  Rank: {}", lora_config.rank);
    println!("  Alpha: {}", lora_config.alpha);
    println!("  Scaling: {:.4}", lora_config.scaling());

    // Apply LoRA to the model
    let lora_model = apply_lora(base_model, &lora_config, &device);

    // Count LoRA parameters (only trainable ones)
    let lora_trainable = count_lora_trainable_params(&lora_model);
    println!("\nLoRA trainable parameters: {}", lora_trainable);
    println!(
        "Parameter reduction: {:.2}%",
        (1.0 - lora_trainable as f64 / base_params as f64) * 100.0
    );

    // Create datasets for training with TUI dashboard
    let train_dataset = SyntheticDataset::new(config.train_size, config.d_input);
    let valid_dataset = SyntheticDataset::new(config.valid_size, config.d_input);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    // Create data loaders with appropriate batchers
    let batcher_train = SyntheticBatcher;
    let batcher_valid = SyntheticBatcher;

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .build(valid_dataset);

    // Setup training with TUI dashboard
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(config.num_epochs)
        .summary();

    println!("\n--- Training with LoRA (TUI Dashboard) ---\n");

    // Run training
    let result = training.launch(Learner::new(
        lora_model,
        AdamConfig::new().init(),
        config.learning_rate,
    ));
    let lora_model = result.model;

    println!("\n--- Training Complete ---\n");

    // Convert test tensors to inner backend (training result is already on InnerBackend)
    let test_input_inner = test_input.inner();
    let test_target_inner = test_target.inner();

    // Evaluate after fine-tuning
    let lora_output = lora_model.forward(test_input_inner.clone());
    let lora_loss: f32 = MseLoss::new()
        .forward(
            lora_output,
            test_target_inner.clone(),
            burn::nn::loss::Reduction::Mean,
        )
        .into_scalar()
        .elem();
    println!("LoRA model loss (after fine-tuning): {:.6}", lora_loss);

    // Merge LoRA weights for inference (model is already on InnerBackend)
    let merged_model = lora_model.clone().merge();
    let merged_output = merged_model.forward(test_input_inner.clone());
    let merged_loss: f32 = MseLoss::new()
        .forward(
            merged_output,
            test_target_inner.clone(),
            burn::nn::loss::Reduction::Mean,
        )
        .into_scalar()
        .elem();
    println!("Merged model loss: {:.6}", merged_loss);

    // Summary
    println!("\n=== Results Summary ===");
    println!("Before fine-tuning: {:.6}", base_loss);
    println!("After fine-tuning:  {:.6}", lora_loss);
    println!("Improvement: {:.2}x", base_loss / lora_loss);
    println!(
        "\nTrained only {} params ({:.2}% of model)",
        lora_trainable,
        lora_trainable as f64 / base_params as f64 * 100.0
    );

    // ====================================
    // Adapter Persistence Demo
    // ====================================
    println!("\n=== Adapter Persistence Demo ===\n");

    // Save adapters to disk (only LoRA weights, not the full model)
    let adapter_path = format!("{artifact_dir}/adapters");
    println!("Saving adapters to: {}", adapter_path);

    // Model is already on InnerBackend after training
    lora_model
        .save_adapters(&adapter_path)
        .expect("Failed to save adapters");

    // Show file sizes to demonstrate efficiency
    let fc1_size = std::fs::metadata(format!("{adapter_path}/fc1.mpk"))
        .map(|m| m.len())
        .unwrap_or(0);
    let fc2_size = std::fs::metadata(format!("{adapter_path}/fc2.mpk"))
        .map(|m| m.len())
        .unwrap_or(0);
    println!("  fc1.mpk: {} bytes", fc1_size);
    println!("  fc2.mpk: {} bytes", fc2_size);
    println!("  Total adapter size: {} bytes", fc1_size + fc2_size);

    // Demonstrate REAL end-to-end loading from disk
    // This is exactly what you would do in production:
    // 1. Load base model from disk
    // 2. Apply LoRA with zero-initialized weights
    // 3. Load trained adapter weights
    println!("\nLoading from disk (full reload)...");

    // Step 1: Create fresh base model structure and load weights from disk
    let fresh_base: SimpleMlp<B::InnerBackend> =
        SimpleMlpConfig::new(config.d_input, config.d_hidden, config.d_output).init(&device);
    let base_record = CompactRecorder::new()
        .load(base_model_path.into(), &device)
        .expect("Failed to load base model");
    let fresh_base = fresh_base.load_record(base_record);

    // Step 2: Apply LoRA (creates zero-initialized A and B matrices)
    let fresh_lora = apply_lora(fresh_base, &lora_config, &device);

    // Step 3: Load trained adapter weights from disk
    let loaded_lora = fresh_lora
        .load_adapters(&adapter_path, &device)
        .expect("Failed to load adapters");

    // Verify loaded model produces same output as the trained model
    let loaded_output = loaded_lora.forward(test_input_inner.clone());
    let loaded_loss: f32 = MseLoss::new()
        .forward(
            loaded_output,
            test_target_inner.clone(),
            burn::nn::loss::Reduction::Mean,
        )
        .into_scalar()
        .elem();
    println!("Loaded model loss: {:.6}", loaded_loss);
    println!(
        "Loss difference vs trained (should be ~0): {:.10}",
        (merged_loss - loaded_loss).abs()
    );

    println!("\n=== Example Complete ===");
}
