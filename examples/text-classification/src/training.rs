use crate::{
    data::{BertCasedTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer},
    model::{TextClassificationModel, TextClassificationModelConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{Module, StateFormat},
    nn::transformer::TransformerEncoderConfig,
    optim::{Sgd, SgdConfig},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

#[derive(Config)]
pub struct ExperimentConfig {
    transformer: TransformerEncoderConfig,
    optimizer: SgdConfig,
    #[config(default = 256)]
    max_seq_length: usize,
    #[config(default = 16)]
    batch_size: usize,
    #[config(default = 10)]
    num_epochs: usize,
}

pub fn train<B: ADBackend, D: TextClassificationDataset + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let dataset_train = Arc::new(dataset_train);
    let dataset_test = Arc::new(dataset_test);
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

    let model = TextClassificationModel::new(&TextClassificationModelConfig::new(
        config.transformer.clone(),
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    ));

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(8)
        .shuffle(42)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(8)
        .build(dataset_test);

    let optim = Sgd::new(&config.optimizer);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer::<burn::tensor::f16>(2, StateFormat::default())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(&format!("{artifact_dir}/config.json")).unwrap();

    model_trained
        .state()
        .convert::<burn::tensor::f16>()
        .save(&format!("{artifact_dir}/model"), &StateFormat::default())
        .unwrap();
}
