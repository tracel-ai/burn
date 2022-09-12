use crate::DatasetIterator;

pub trait Dataset<I>: Send + Sync {
    fn iter(&self) -> DatasetIterator<'_, I>;
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
