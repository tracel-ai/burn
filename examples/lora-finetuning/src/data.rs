//! Synthetic dataset for LoRA fine-tuning example.
//!
//! This module provides a simple synthetic dataset for demonstrating LoRA training.
//! The task is to predict sin(mean(x) * pi) for input vector x.

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use rand::Rng;

/// A single synthetic training item.
#[derive(Clone, Debug)]
pub struct SyntheticItem {
    /// Input features.
    pub input: Vec<f32>,
    /// Target value: sin(mean(input) * pi).
    pub target: f32,
}

/// Batched data for training.
#[derive(Clone, Debug)]
pub struct SyntheticBatch<B: Backend> {
    /// Input tensor with shape [batch_size, d_input].
    pub inputs: Tensor<B, 2>,
    /// Target tensor with shape [batch_size, 1].
    pub targets: Tensor<B, 2>,
}

/// Batcher for synthetic data.
#[derive(Clone, Default)]
pub struct SyntheticBatcher;

impl<B: Backend> Batcher<B, SyntheticItem, SyntheticBatch<B>> for SyntheticBatcher {
    fn batch(&self, items: Vec<SyntheticItem>, device: &B::Device) -> SyntheticBatch<B> {
        let inputs: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.input.as_slice(), device).unsqueeze())
            .collect();

        let targets: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.target], device).unsqueeze())
            .collect();

        SyntheticBatch {
            inputs: Tensor::cat(inputs, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

/// In-memory synthetic dataset.
pub struct SyntheticDataset {
    items: Vec<SyntheticItem>,
}

impl Dataset<SyntheticItem> for SyntheticDataset {
    fn get(&self, index: usize) -> Option<SyntheticItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl SyntheticDataset {
    /// Generate synthetic dataset.
    ///
    /// Task: predict sin(mean(x) * pi) for input vector x.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate.
    /// * `d_input` - Dimension of input features.
    pub fn new(size: usize, d_input: usize) -> Self {
        let mut rng = rand::rng();

        let items = (0..size)
            .map(|_| {
                let input: Vec<f32> = (0..d_input).map(|_| rng.random_range(-1.0..1.0)).collect();
                let mean: f32 = input.iter().sum::<f32>() / d_input as f32;
                let target = (mean * std::f32::consts::PI).sin();
                SyntheticItem { input, target }
            })
            .collect();

        Self { items }
    }
}
