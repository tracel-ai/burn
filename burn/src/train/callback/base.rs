use crate::data::dataloader::Progress;

pub trait TrainerCallback<T, V>: Send {
    fn on_train_item(&mut self, _item: T) {}
    fn on_valid_item(&mut self, _item: V) {}
    fn on_train_end_epoch(&mut self) {}
    fn on_valid_end_epoch(&mut self) {}
}

#[derive(new)]
pub struct TrainerItem<T> {
    pub item: T,
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
}
