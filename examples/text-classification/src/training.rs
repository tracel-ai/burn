// This module trains a text classification model using the provided training and testing datasets,
// as well as the provided configuration. It first initializes a tokenizer and batchers for the datasets,
// then initializes the model and data loaders for the datasets. The function then initializes
// an optimizer and a learning rate scheduler, and uses them along with the model and datasets
// to build a learner, which is used to train the model. The trained model and the configuration are
// then saved to the specified directory.

use crate::{
    data::{BertCasedTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer},
    model::TextClassificationModelConfig,
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLRSchedulerConfig,
    module::Module,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

// Define configuration struct for the experiment
#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 256)]
    pub max_seq_length: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub num_epochs: usize,
}

// Define train function
pub fn train<B: ADBackend, D: TextClassificationDataset + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,  // Training dataset
    dataset_test: D,   // Testing dataset
    config: ExperimentConfig, // Experiment configuration
    artifact_dir: &str, // Directory to save model and config files
) {
    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Initialize batchers for training and testing data
    let batcher_train = TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    );
    let batcher_test = TextClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    );

    // Initialize model
    let model = TextClassificationModelConfig::new(
        config.transformer.clone(),
        D::num_classes(),
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init();

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 50_000));
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 5_000));

    // Initialize optimizer
    let optim = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = NoamLRSchedulerConfig::new(0.25)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .metric_train_plot(LearningRateMetric::new())
        .with_file_checkpointer(2, CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
