use std::usize;

use crate::Dataset;

#[derive(new)]
pub struct TimeSeriesDataset<D> {
    dataset: D,
    window_size: usize,
}

impl<D, I> Dataset<Vec<I>> for TimeSeriesDataset<D>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<Vec<I>> {
        (index..index + self.window_size)
            .map(|x| self.dataset.get(x))
            .collect()
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.window_size
    }
}
