use crate::{Dataset, DatasetIterator};

pub struct ComposedDataset<I> {
    datasets: Vec<Box<dyn Dataset<I>>>,
}

impl<I> ComposedDataset<I> {
    pub fn new(datasets: Vec<Box<dyn Dataset<I>>>) -> Self {
        Self { datasets }
    }
}

impl<I> Dataset<I> for ComposedDataset<I>
where
    I: Clone,
{
    fn get(&self, index: usize) -> Option<I> {
        let mut current_index = 0;
        for dataset in self.datasets.iter() {
            if index < dataset.len() + current_index {
                return dataset.get(index - current_index);
            }
            current_index += dataset.len();
        }
        None
    }
    fn iter(&self) -> DatasetIterator<'_, I> {
        DatasetIterator::new(self)
    }
    fn len(&self) -> usize {
        let mut total = 0;
        for dataset in self.datasets.iter() {
            total += dataset.len();
        }
        total
    }

    fn is_empty(&self) -> bool {
        let mut is_empty = true;

        for dataset in self.datasets.iter() {
            if !dataset.is_empty() {
                is_empty = false;
            }
        }

        is_empty
    }
}
