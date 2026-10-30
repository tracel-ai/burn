use std::sync::Arc;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            Dataset,
            transform::{SamplerDataset, SamplerDatasetOptions},
        },
    },
    module::parallel::{DistributedLayeredModel, LayerPlacement},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
};
use text_classification::{
    DbPediaDataset, TextClassificationDataset,
    data::{
        BertCasedTokenizer, TextClassificationBatcher, TextClassificationItem,
        TextClassificationTrainingBatch, Tokenizer,
    },
    training::ExperimentConfig,
};

use crate::{
    inference::correct_count,
    layered::{LayeredClassifier, Tokens},
};

const BATCH: usize = 32;
const STEPS: usize = 300;
const EVAL: usize = 512;
const LEARNING_RATE: f64 = 3e-4;
/// Fixes the head's initial weights and the samples drawn, so runs on different placements compare.
const SEED: u64 = 0;
const LOG_EVERY: usize = 25;

/// Fine-tune the AG News classifier `text-classification` trained on one device for DbPedia,
/// split across `devices`: its embeddings and encoder layers load straight onto their devices, the
/// head starts afresh for DbPedia's classes, and every parameter trains on the device it lives on.
pub fn finetune(devices: Vec<Device>, artifact_dir: &str) {
    assert!(!devices.is_empty(), "no device to place the model on");
    // Every device needs autodiff, since a gradient has to survive the hop back to the layer before.
    let devices: Vec<Device> = devices.into_iter().map(Device::autodiff).collect();
    for device in &devices {
        device.seed(SEED);
    }
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json"))
        .expect("the artifact directory should hold ag-news-train's config");
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    let placement = LayerPlacement::even(&devices, config.transformer.n_layers);
    let model = LayeredClassifier::new(
        &config.transformer,
        config.seq_length,
        tokenizer.vocab_size(),
        DbPediaDataset::num_classes(),
        &placement,
    );
    let mut model = DistributedLayeredModel::new(model, &placement);
    // AG News's head has its own classes, so only the embeddings and the encoder layers load.
    let mut store = LayeredClassifier::remap_from_single_device(BurnpackStore::from_file(format!(
        "{artifact_dir}/model"
    )))
    .with_regex(r"^(input|hidden)\.")
    .allow_partial(true);
    model
        .load_from(&mut store)
        .expect("the checkpoint should cover the embeddings and every encoder layer");
    model.model().print_placement(model.placement());

    let batcher = TextClassificationBatcher::new(tokenizer, config.seq_length);
    let eval = sample(DbPediaDataset::test(), EVAL);
    println!(
        "test accuracy before: {:.3}",
        accuracy(&model, &batcher, &eval)
    );

    // DbPedia's splits are ordered by class, so batches are drawn at random.
    let train = SamplerDataset::new(DbPediaDataset::train(), seeded(STEPS * BATCH));
    let mut optimizer = AdamConfig::new().init();
    for step in 1..=STEPS {
        let items = (step * BATCH - BATCH..step * BATCH)
            .filter_map(|index| train.get(index).ok())
            .collect();
        let batch: TextClassificationTrainingBatch = batcher.batch(items, &placement.input);
        let logits = model.forward(Tokens {
            ids: batch.tokens,
            mask_pad: batch.mask_pad,
        });
        let labels = batch.labels.to_device(&logits.device());
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, labels);
        if step == 1 || step % LOG_EVERY == 0 {
            println!(
                "step {step:>3}: loss {:.4}",
                loss.clone().into_scalar::<f32>()
            );
        }

        // Each parameter is updated on its own device: the optimizer follows the gradients.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }

    println!(
        "test accuracy after:  {:.3}",
        accuracy(&model, &batcher, &eval)
    );
}

/// `count` items of `dataset` drawn at random, held so every evaluation reads the same ones.
fn sample(dataset: DbPediaDataset, count: usize) -> Vec<TextClassificationItem> {
    let sampled = SamplerDataset::new(dataset, seeded(count));
    (0..count)
        .filter_map(|index| sampled.get(index).ok())
        .collect()
}

/// `count` samples, drawn in the same order on every run.
fn seeded(count: usize) -> SamplerDatasetOptions {
    SamplerDatasetOptions::default()
        .with_fixed_size(count)
        .with_seed(SEED)
}

fn accuracy(
    model: &DistributedLayeredModel<LayeredClassifier>,
    batcher: &TextClassificationBatcher,
    items: &[TextClassificationItem],
) -> f32 {
    let model = model.valid();
    let device = &model.placement().input;
    let correct: usize = items
        .chunks(BATCH)
        .map(|chunk| {
            let batch: TextClassificationTrainingBatch = batcher.batch(chunk.to_vec(), device);
            let logits = model.forward(Tokens {
                ids: batch.tokens,
                mask_pad: batch.mask_pad,
            });
            correct_count(logits, batch.labels)
        })
        .sum();
    correct as f32 / items.len() as f32
}
