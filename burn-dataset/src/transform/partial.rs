use crate::{Dataset, DatasetIterator};
use std::sync::Arc;

pub struct PartialDataset<I> {
    dataset: Arc<dyn Dataset<I>>,
    start_index: usize,
    end_index: usize,
}

impl<I> PartialDataset<I> {
    pub fn new(dataset: Arc<dyn Dataset<I>>, start_index: usize, end_index: usize) -> Self {
        Self {
            dataset,
            start_index,
            end_index,
        }
    }
    pub fn split(dataset: Arc<dyn Dataset<I>>, num: usize) -> Vec<PartialDataset<I>> {
        let num_batch = dataset.len() / num;
        let mut current = 0;
        let mut datasets = Vec::with_capacity(num);

        for _ in 0..num {
            let dataset = PartialDataset::new(dataset.clone(), current, current + num_batch);
            current += num_batch;
            datasets.push(dataset);
        }

        datasets
    }
}

impl<I> Dataset<I> for PartialDataset<I>
where
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = index + self.start_index;
        if index < self.start_index && index >= self.end_index {
            return None;
        }
        self.dataset.get(index)
    }

    fn iter<'a>(&'a self) -> DatasetIterator<'a, I> {
        DatasetIterator::new(self)
    }
    fn len(&self) -> usize {
        usize::min(self.end_index - self.start_index, self.dataset.len())
    }
}
