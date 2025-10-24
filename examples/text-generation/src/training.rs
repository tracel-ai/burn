use crate::{
    data::{Gpt2Tokenizer, TextGenerationBatcher, TextGenerationItem, Tokenizer},
    model::TextGenerationModelConfig,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, transform::SamplerDataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, LearningStrategy,
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};
use std::sync::Arc;

#[derive(Config, Debug)]
pub struct ExperimentConfig {
    transformer: TransformerEncoderConfig,
    optimizer: AdamConfig,
    #[config(default = 512)]
    max_seq_length: usize,
    #[config(default = 6)]
    batch_size: usize,
    #[config(default = 50)]
    num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(Gpt2Tokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        config.max_seq_length,
    )
    .init::<B>(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000));

    let accum = 6; // Effective batch size = 6 * 6 = 32.
    let optim = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(config.transformer.d_model)
        .init()
        .unwrap();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(PerplexityMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            optim,
            lr_scheduler,
            LearningStrategy::SingleDevice(device.clone()),
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            model_trained.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
