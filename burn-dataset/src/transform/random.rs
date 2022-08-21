use crate::{Dataset, DatasetIterator};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct ShuffledDataset<I> {
    dataset: Box<dyn Dataset<I>>,
    indexes: Vec<usize>,
}

impl<I> ShuffledDataset<I> {
    pub fn new(dataset: Box<dyn Dataset<I>>) -> Self {
        let mut indexes = Vec::with_capacity(dataset.len());
        for i in 0..dataset.len() {
            indexes.push(i);
        }
        let mut rng = thread_rng();
        indexes.shuffle(&mut rng);

        Self { dataset, indexes }
    }
}

impl<I> Dataset<I> for ShuffledDataset<I>
where
    I: Clone,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = match self.indexes.get(index) {
            Some(index) => index,
            None => return None,
        };
        match self.dataset.get(*index) {
            Some(item) => Some(item.clone()),
            None => None,
        }
    }
    fn iter<'a>(&'a self) -> DatasetIterator<'a, I> {
        DatasetIterator::new(self)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
