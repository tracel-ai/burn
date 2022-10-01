pub trait TrainStep {
    type Input;
    type Output;

    fn step(&mut self, item: Self::Input) -> Self::Output;
}

pub trait ValidStep {
    type Input;
    type Output;

    fn step(&self, item: Self::Input) -> Self::Output;
}

pub trait CheckpointModel {
    fn checkpoint(&self, epoch: usize);
    fn load_checkpoint(&mut self, epoch: usize);
}
