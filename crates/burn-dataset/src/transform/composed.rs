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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_composed_dataset() {
        let dataset1 = FakeDataset::<String>::new(10);
        let dataset2 = FakeDataset::<String>::new(5);

        let items1 = dataset1.iter().collect::<Vec<_>>();
        let items2 = dataset2.iter().collect::<Vec<_>>();

        let composed = ComposedDataset::new(vec![dataset1, dataset2]);

        assert_eq!(composed.len(), 15);

        let expected_items: Vec<String> = items1.iter().chain(items2.iter()).cloned().collect();

        let items = composed.iter().collect::<Vec<_>>();

        assert_eq!(items, expected_items);
    }
}
