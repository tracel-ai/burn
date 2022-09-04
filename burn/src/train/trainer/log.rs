use crate::data::dataloader::Progress;

pub struct TrainerItem<T> {
    pub progress: Progress,
    pub item: T,
    pub epoch: Option<usize>,
    pub epoch_total: Option<usize>,
    pub iteration: Option<usize>,
}

impl<T> TrainerItem<T> {
    pub fn new(item: T, progress: Progress) -> Self {
        Self {
            epoch: None,
            epoch_total: None,
            iteration: None,
            progress,
            item,
        }
    }

    pub fn iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
        self
    }

    pub fn epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }

    pub fn epoch_total(mut self, epoch: usize) -> Self {
        self.epoch_total = Some(epoch);
        self
    }
}
