use super::{LrScheduler, LrSchedulerRecord};
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
    fn step(&mut self) -> LearningRate {
        self.lr
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(self, _record: LrSchedulerRecord) -> Self {
        self
    }
}

impl LrScheduler for LearningRate {
    fn step(&mut self) -> LearningRate {
        *self
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(self, _record: LrSchedulerRecord) -> Self {
        self
    }
}
