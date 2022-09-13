pub use crate::data::dataset::{Dataset, DatasetIterator};
use std::iter::Iterator;

#[derive(Clone, Debug)]
pub struct Progress {
    pub items_processed: usize,
    pub items_total: usize,
}

pub trait DataLoaderIterator<O>: Iterator<Item = O> {
    fn progress(&self) -> Progress;
}

pub trait DataLoader<O> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a>;
}
