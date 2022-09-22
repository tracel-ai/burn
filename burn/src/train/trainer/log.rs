use crate::data::dataloader::Progress;

#[derive(new)]
pub struct TrainerItem<T> {
    pub item: T,
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
}
