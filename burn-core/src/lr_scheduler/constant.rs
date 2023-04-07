use super::LearningRateScheduler;
use crate::LearningRate;

/// Constant learning rate implementing [learning rate scheduler](LearningRateScheduler).
#[derive(new, Clone, Debug)]
pub struct ConstantLearningRate {
    learning_rate: LearningRate,
}

impl LearningRateScheduler for ConstantLearningRate {
    type Record = ();

    fn step(&mut self) -> LearningRate {
        self.learning_rate
    }

    fn to_record(&self) -> Self::Record {
        ()
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }
}
