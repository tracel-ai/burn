use crate::Dataset;
use std::error::Error;

/// Compose multiple datasets together to create a bigger one.
#[derive(new)]
pub struct ComposedDataset<D> {
    datasets: Vec<D>,
}

impl<D, I, E> Dataset<I, E> for ComposedDataset<D>
where
    D: Dataset<I, E>,
    I: Clone,
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E> {
        let mut current_index = 0;
        for dataset in self.datasets.iter() {
            if index < dataset.len() + current_index {
                return dataset.get(index - current_index);
            }
            current_index += dataset.len();
        }
        panic!(
            "Index out of bounds for ComposedDataset: {index} >= {}",
            self.len()
        );
    }
    fn len(&self) -> usize {
        let mut total = 0;
        for dataset in self.datasets.iter() {
            total += dataset.len();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_composed_dataset() {
        let dataset1 = FakeDataset::<String>::new(10);
        let dataset2 = FakeDataset::<String>::new(5);

        let items1 = dataset1.iter().map(Result::unwrap).collect::<Vec<_>>();
        let items2 = dataset2.iter().map(Result::unwrap).collect::<Vec<_>>();

        let composed = ComposedDataset::new(vec![dataset1, dataset2]);

        assert_eq!(composed.len(), 15);

        let expected_items: Vec<String> = items1.iter().chain(items2.iter()).cloned().collect();

        let items = composed.iter().map(Result::unwrap).collect::<Vec<_>>();

        assert_eq!(items, expected_items);
    }
}
