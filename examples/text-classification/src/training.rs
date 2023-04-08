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
    record::{DefaultRecordSettings, Record},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

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

pub fn train<B: ADBackend, D: TextClassificationDataset + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let dataset_train = Arc::new(SamplerDataset::new(Box::new(dataset_train), 50_000));
    let dataset_test = Arc::new(SamplerDataset::new(Box::new(dataset_test), 5_000));
    let n_classes = D::num_classes();

    let tokenizer = Arc::new(BertCasedTokenizer::default());
    let batcher_train = Arc::new(TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));
    let batcher_test = Arc::new(TextClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));

    let model = TextClassificationModelConfig::new(
        config.transformer.clone(),
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(dataset_test);

    let optim = config.optimizer.init();
    let lr_scheduler = NoamLRSchedulerConfig::new(0.25)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .metric_train_plot(LearningRateMetric::new())
        .with_file_checkpointer::<DefaultRecordSettings>(2)
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(&format!("{artifact_dir}/config.json")).unwrap();

    model_trained
        .into_record()
        .record::<DefaultRecordSettings>(format!("{artifact_dir}/model").into())
        .unwrap();
}
