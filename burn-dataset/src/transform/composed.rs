use crate::Dataset;

/// Compose multiple datasets together to create a bigger one.
#[derive(new)]
pub struct ComposedDataset<D> {
    datasets: Vec<D>,
}

impl<D, I> Dataset<I> for ComposedDataset<D>
where
    D: Dataset<I>,
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
    fn len(&self) -> usize {
        let mut total = 0;
        for dataset in self.datasets.iter() {
            total += dataset.len();
        }
        total
    }
}
