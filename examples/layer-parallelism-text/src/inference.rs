use std::sync::Arc;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::parallel::{DistributedLayeredModel, LayerPlacement},
    prelude::*,
    store::{BurnpackStore, ModuleRecord, ModuleSnapshot},
    tensor::Tolerance,
};
use text_classification::{
    AgNewsDataset, TextClassificationDataset,
    data::{
        BertCasedTokenizer, TextClassificationBatcher, TextClassificationTrainingBatch, Tokenizer,
    },
    model::{TextClassificationModel, TextClassificationModelConfig},
    training::ExperimentConfig,
};

use crate::layered::{LayeredClassifier, Tokens};

const SAMPLES: usize = 512;
const BATCH: usize = 32;

/// Split the AG News classifier `text-classification` trained on one device across `devices`,
/// load its checkpoint straight onto them, and check the split against the single-device model on
/// the test set.
pub fn infer(devices: Vec<Device>, artifact_dir: &str) {
    assert!(!devices.is_empty(), "no device to place the model on");
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json"))
        .expect("the artifact directory should hold ag-news-train's config");
    let tokenizer = Arc::new(BertCasedTokenizer::default());
    let n_classes = AgNewsDataset::num_classes();

    let placement = LayerPlacement::even(&devices, config.transformer.n_layers);
    let model = LayeredClassifier::new(
        &config.transformer,
        config.seq_length,
        tokenizer.vocab_size(),
        n_classes,
        &placement,
    );
    let mut split = DistributedLayeredModel::new(model, &placement);
    let mut store = LayeredClassifier::remap_from_single_device(BurnpackStore::from_file(format!(
        "{artifact_dir}/model"
    )));
    split
        .load_from(&mut store)
        .expect("the checkpoint should cover every layer");
    split.model().print_placement(split.placement());

    let single = TextClassificationModelConfig::new(
        config.transformer.clone(),
        n_classes,
        tokenizer.vocab_size(),
        config.seq_length,
    )
    .init(&placement.input)
    .load_record(
        ModuleRecord::load(format!("{artifact_dir}/model"))
            .expect("the artifact directory should hold ag-news-train's model"),
    );

    let batcher = TextClassificationBatcher::new(tokenizer, config.seq_length);
    let test = AgNewsDataset::test();
    let mut correct = 0;
    for start in (0..SAMPLES.min(test.len())).step_by(BATCH) {
        let items = (start..(start + BATCH).min(test.len()))
            .filter_map(|index| test.get(index).ok())
            .collect();
        let batch: TextClassificationTrainingBatch = batcher.batch(items, &placement.input);
        let logits = split.forward(Tokens {
            ids: batch.tokens.clone(),
            mask_pad: batch.mask_pad.clone(),
        });
        correct += correct_count(logits.clone(), batch.labels.clone());
        assert_matches_single_device(logits, &single, batch);
    }
    println!(
        "the split matches the single-device model, {correct} of {} test samples right",
        SAMPLES.min(test.len())
    );
}

fn assert_matches_single_device(
    logits: Tensor<2>,
    single: &TextClassificationModel,
    batch: TextClassificationTrainingBatch,
) {
    let expected = single.forward(batch).output;
    logits
        .to_device(&expected.device())
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::permissive());
}

/// How many rows of `logits` pick the class `labels` holds.
pub fn correct_count(logits: Tensor<2>, labels: Tensor<1, Int>) -> usize {
    let predicted = logits.argmax(1).squeeze_dim::<1>(1);
    let labels = labels.to_device(&predicted.device());
    predicted.equal(labels).int().sum().into_scalar::<i64>() as usize
}
