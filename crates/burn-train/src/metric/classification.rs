use burn_core::prelude::{Backend, Bool, Tensor};
use std::num::NonZeroUsize;

/// The reduction strategy for classification metrics.
#[derive(Copy, Clone, Default)]
pub enum ClassReduction {
    /// Computes the statistics over all classes before averaging
    Micro,
    /// Computes the statistics independently for each class before averaging
    #[default]
    Macro,
}

/// Input for classification metrics
#[derive(new, Debug, Clone)]
pub struct ClassificationInput<B: Backend> {
    /// Sample x Class Non thresholded normalized predictions.
    pub predictions: Tensor<B, 2>,
    /// Sample x Class one-hot encoded target.
    pub targets: Tensor<B, 2, Bool>,
}

impl<B: Backend> From<ClassificationInput<B>> for (Tensor<B, 2>, Tensor<B, 2, Bool>) {
    fn from(input: ClassificationInput<B>) -> Self {
        (input.predictions, input.targets)
    }
}

impl<B: Backend> From<(Tensor<B, 2>, Tensor<B, 2, Bool>)> for ClassificationInput<B> {
    fn from(value: (Tensor<B, 2>, Tensor<B, 2, Bool>)) -> Self {
        Self::new(value.0, value.1)
    }
}

pub enum ClassificationConfig {
    Binary {
        threshold: f64,
        class_reduction: ClassReduction,
    },
    Multiclass {
        top_k: NonZeroUsize,
        class_reduction: ClassReduction,
    },
    Multilabel {
        threshold: f64,
        class_reduction: ClassReduction,
    },
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self::Binary {
            threshold: 0.5,
            class_reduction: Default::default(),
        }
    }
}
