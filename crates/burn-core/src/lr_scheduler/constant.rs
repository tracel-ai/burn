use burn_tensor::backend::Backend;

use super::LrScheduler;
use crate::LearningRate;

/// Constant learning rate implementing [learning rate scheduler](LrScheduler).
///
/// # Notes
///
/// You can also use [learning rate](LearningRate) which the same effect.
#[derive(new, Clone, Debug)]
pub struct ConstantLr {
    lr: LearningRate,
}

impl From<LearningRate> for ConstantLr {
    fn from(lr: LearningRate) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantLr {
    type Record<B: Backend> = ();

    fn step(&mut self) -> LearningRate {
        self.lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {}

    fn load_record<B: Backend>(self, _record: Self::Record<B>) -> Self {
        self
    }
}

impl LrScheduler for LearningRate {
    type Record<B: Backend> = ();

    fn step(&mut self) -> LearningRate {
        *self
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {}

    fn load_record<B: Backend>(self, _record: Self::Record<B>) -> Self {
        self
    }
}
