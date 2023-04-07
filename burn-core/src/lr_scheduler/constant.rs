use super::{LearningRate, LearningRateScheduler};

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
}
