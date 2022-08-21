pub use crate::data::dataset::{Dataset, DatasetIterator};
use std::iter::Iterator;

pub trait DataLoader<O> {
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = O> + 'a>;
    fn len(&self) -> usize;
}
