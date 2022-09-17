use crate::DatasetIterator;

pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}
