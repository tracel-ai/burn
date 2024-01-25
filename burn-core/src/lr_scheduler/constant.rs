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

impl<B: Backend> LrScheduler<B> for ConstantLr {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        self.lr
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}

impl<B: Backend> LrScheduler<B> for LearningRate {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        *self
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
