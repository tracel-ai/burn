use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    prelude::*,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// Dataset parameters
pub const NUM_SEQUENCES: usize = 1000;
pub const SEQ_LENGTH: usize = 10;
pub const NOISE_LEVEL: f32 = 0.1;
pub const RANDOM_SEED: u64 = 5;

// Generate a sequence where each number is the sum of previous two numbers plus noise
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceDatasetItem {
    pub sequence: Vec<f32>,
    pub target: f32,
}

impl SequenceDatasetItem {
    pub fn new(seq_length: usize, noise_level: f32) -> Self {
        // Start with two random numbers between 0 and 1
        let mut seq = vec![rand::thread_rng().gen(), rand::thread_rng().gen()];

        // Generate sequence
        for _i in 0..seq_length {
            // Next number is sum of previous two plus noise
            let normal = Normal::new(0.0, noise_level).unwrap();
            let next_val =
                seq[seq.len() - 2] + seq[seq.len() - 1] + normal.sample(&mut rand::thread_rng());
            seq.push(next_val);
        }

        Self {
            // Convert to sequence and target
            sequence: seq[0..seq.len() - 1].to_vec(), // All but last
            target: seq[seq.len() - 1],               // Last value
        }
    }
}

// Custom Dataset for Sequence Data
pub struct SequenceDataset {
    dataset: InMemDataset<SequenceDatasetItem>,
}

impl SequenceDataset {
    pub fn new(num_sequences: usize, seq_length: usize, noise_level: f32) -> Self {
        let mut items = vec![];
        for _i in 0..num_sequences {
            items.push(SequenceDatasetItem::new(seq_length, noise_level));
        }
        let dataset = InMemDataset::new(items);

        Self { dataset }
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone, Debug)]
pub struct SequenceBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct SequenceBatch<B: Backend> {
    pub sequences: Tensor<B, 3>, // [batch_size, seq_length, input_size]
    pub targets: Tensor<B, 2>,   // [batch_size, 1]
}

impl<B: Backend> SequenceBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<SequenceDatasetItem, SequenceBatch<B>> for SequenceBatcher<B> {
    fn batch(&self, items: Vec<SequenceDatasetItem>) -> SequenceBatch<B> {
        let mut sequences: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let seq_tensor = Tensor::<B, 1>::from_floats(item.sequence.as_slice(), &self.device);
            // Add feature dimension, the input_size is 1 implicitly. We can change the input_size here with some operations
            sequences.push(seq_tensor.unsqueeze_dims(&[-1]));
        }
        let sequences = Tensor::stack(sequences, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.target], &self.device))
            .collect();
        let targets = Tensor::stack(targets, 0);

        SequenceBatch { sequences, targets }
    }
}
