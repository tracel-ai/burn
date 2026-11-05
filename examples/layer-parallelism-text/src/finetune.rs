use std::sync::Arc;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            Dataset,
            transform::{SamplerDataset, SamplerDatasetOptions},
        },
    },
    module::parallel::{DistributedLayeredModel, LayerPlacement, PipelineSchedule},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, parallel::TrainingPipeline},
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
/// Each step's batch is cut into this many microbatches, which the stages overlap.
const MICROBATCHES: usize = 4;
const SCHEDULE: PipelineSchedule = PipelineSchedule::OneForwardOneBackward;
const STEPS: usize = 300;
const EVAL: usize = 512;
const LEARNING_RATE: f64 = 3e-4;
/// Fixes the head's initial weights and the samples drawn, so runs on different placements compare.
const SEED: u64 = 0;
const LOG_EVERY: usize = 25;

/// Fine-tune the AG News classifier `text-classification` trained on one device for DbPedia,
/// split across `devices`: its embeddings and encoder layers load straight onto their devices, the
/// head starts afresh for DbPedia's classes, and it trains as a pipeline: each stage runs its
/// microbatches' passes on a thread of its own, and every parameter is updated on its device.
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
    let mut training = TrainingPipeline::new(model, SCHEDULE, AdamConfig::new().init());
    println!(
        "test accuracy before: {:.3}",
        accuracy(&training, &batcher, &eval, &placement.input)
    );

    // DbPedia's splits are ordered by class, so batches are drawn at random.
    let train = SamplerDataset::new(DbPediaDataset::train(), seeded(STEPS * BATCH));
    for step in 1..=STEPS {
        let items = (step * BATCH - BATCH..step * BATCH)
            .map(|index| {
                train
                    .get(index)
                    .expect("the sampler draws an item for every index of every step")
            })
            .collect();
        let batch: TextClassificationTrainingBatch = batcher.batch(items, &placement.input);
        let losses = training.step(
            microbatches(batch.tokens, batch.mask_pad),
            batch.labels.chunk(MICROBATCHES, 0),
            |logits, labels| {
                let labels = labels.to_device(&logits.device());
                CrossEntropyLossConfig::new()
                    .init(&logits.device())
                    .forward(logits, labels)
            },
            LEARNING_RATE,
        );
        if step == 1 || step % LOG_EVERY == 0 {
            let loss = Tensor::cat(losses, 0).mean().into_scalar::<f32>();
            println!("step {step:>3}: loss {loss:.4}");
        }
    }

    println!(
        "test accuracy after:  {:.3}",
        accuracy(&training, &batcher, &eval, &placement.input)
    );
}

/// A batch's tokens cut into `MICROBATCHES` inputs along the batch.
fn microbatches(ids: Tensor<2, Int>, mask_pad: Tensor<2, Bool>) -> Vec<Tokens> {
    ids.chunk(MICROBATCHES, 0)
        .into_iter()
        .zip(mask_pad.chunk(MICROBATCHES, 0))
        .map(|(ids, mask_pad)| Tokens { ids, mask_pad })
        .collect()
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
    training: &TrainingPipeline<LayeredClassifier>,
    batcher: &TextClassificationBatcher,
    items: &[TextClassificationItem],
    device: &Device,
) -> f32 {
    let batches: Vec<TextClassificationTrainingBatch> = items
        .chunks(BATCH)
        .map(|chunk| batcher.batch(chunk.to_vec(), device))
        .collect();
    let inputs = batches
        .iter()
        .map(|batch| Tokens {
            ids: batch.tokens.clone(),
            mask_pad: batch.mask_pad.clone(),
        })
        .collect();
    let correct: usize = training
        .forward(inputs)
        .into_iter()
        .zip(batches)
        .map(|(logits, batch)| correct_count(logits, batch.labels))
        .sum();
    correct as f32 / items.len() as f32
}
