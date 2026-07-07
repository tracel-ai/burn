use std::sync::Arc;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::module::LoraConfig;
use burn::train::metric::{
    AccuracyMetric, CudaMetric, IterationSpeedMetric, LearningRateMetric, LossMetric,
};
use burn::train::{ExecutionStrategy, Learner, SupervisedTraining};
use burn::{module::LoraMapper, prelude::*};

use crate::{
    TextClassificationDataset,
    data::{BertCasedTokenizer, TextClassificationBatcher, Tokenizer},
    model::TextClassificationModelConfig,
    training::{ExperimentConfig, create_artifact_dir},
};

// Finetune pre-trained weights using LoRA.
pub fn lora_finetuning<D: TextClassificationDataset + 'static>(
    strategy: ExecutionStrategy,
    dataset_train: D,         // Training dataset
    dataset_test: D,          // Testing dataset
    config: ExperimentConfig, // Experiment configuration
    artifact_dir: &str,       // Directory to save model and config files
    num_class_before: usize,  // The number of class of the pretrained model
) {
    create_artifact_dir(artifact_dir);

    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Initialize batcher
    let batcher = TextClassificationBatcher::new(tokenizer.clone(), config.seq_length);

    // Initialize model
    let model = TextClassificationModelConfig::new(
        config.transformer.clone(),
        num_class_before,
        tokenizer.vocab_size(),
        config.seq_length,
    )
    .init(&strategy.main_device().clone().autodiff());

    // Load pre-trained weights.
    let model = model.load_file(&"model.bpk");

    // Apply LoRA to the attention module's query, value, output and feed-forward weights.
    let r = 8.0;
    let mut model = model.apply_lora(&mut LoraMapper::new(LoraConfig::new(r as usize, 2.0 * r)));
    // Reset the classification head with the current dataset's number of classes.
    model.reset_head(D::num_classes());

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_train, 25_000));
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_test, 2500));

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = 1e-3;

    // Initialize learner
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train(IterationSpeedMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_default_checkpointers()
        .with_training_strategy(strategy.into())
        .num_epochs(config.num_epochs)
        .summary();

    // Train the model
    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    result
        .model
        .into_record()
        .save(format!("{artifact_dir}/model"))
        .unwrap();
}
