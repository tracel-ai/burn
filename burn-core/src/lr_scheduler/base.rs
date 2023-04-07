use crate::record::Record;

pub type LearningRate = f64;

pub trait LearningRateScheduler {
    type Record: Record;

    fn step(&mut self) -> LearningRate;
    fn to_record(&self) -> Self::Record;
}
