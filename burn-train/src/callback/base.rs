use burn_core::{data::dataloader::Progress, LearningRate};

pub trait LearnerCallback<T, V>: Send {
    fn on_train_item(&mut self, _item: LearnerItem<T>) {}
    fn on_valid_item(&mut self, _item: LearnerItem<V>) {}
    fn on_train_end_epoch(&mut self, _epoch: usize) {}
    fn on_valid_end_epoch(&mut self, _epoch: usize) {}
}

#[derive(new)]
pub struct LearnerItem<T> {
    pub item: T,
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
    pub lr: Option<LearningRate>,
}
