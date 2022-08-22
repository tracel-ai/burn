use crate::{Dataset, DatasetIterator};
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::sync::Arc;

pub struct ShuffledDataset<I> {
    dataset: Arc<dyn Dataset<I>>,
    indexes: Vec<usize>,
}

impl<I> ShuffledDataset<I> {
    pub fn new(dataset: Arc<dyn Dataset<I>>, rng: &mut StdRng) -> Self {
        let mut indexes = Vec::with_capacity(dataset.len());
        for i in 0..dataset.len() {
            indexes.push(i);
        }
        indexes.shuffle(rng);

        Self { dataset, indexes }
    }

    pub fn with_seed(dataset: Arc<dyn Dataset<I>>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(dataset, &mut rng)
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
