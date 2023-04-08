use super::LRScheduler;
use crate::LearningRate;

/// Constant learning rate implementing [learning rate scheduler](LRScheduler).
///
/// # Notes
///
/// You can also use [learning rate](LearningRate) which the same effect.
#[derive(new, Clone, Debug)]
pub struct ConstantLR {
    lr: LearningRate,
}

impl From<LearningRate> for ConstantLR {
    fn from(lr: LearningRate) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        self.lr
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}

impl LRScheduler for LearningRate {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        *self
    }

    fn to_record(&self) -> Self::Record {}

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
