use crate::Dataset;
use std::{marker::PhantomData, sync::Arc};

/// Only use a fraction of an existing dataset lazily.
#[derive(new)]
pub struct PartialDataset<D, I> {
    dataset: D,
    start_index: usize,
    end_index: usize,
    input: PhantomData<I>,
}

impl<D, I> PartialDataset<D, I>
where
    D: Dataset<I>,
{
    /// Splits a dataset into multiple partial datasets.
    pub fn split(dataset: D, num: usize) -> Vec<PartialDataset<Arc<D>, I>> {
        let dataset = Arc::new(dataset); // cheap cloning.

        let mut current = 0;
        let mut datasets = Vec::with_capacity(num);

        let batch_size = dataset.len() / num;

        for i in 0..num {
            let start = current;
            let mut end = current + batch_size;

            if i == (num - 1) {
                end = dataset.len();
            }

            let dataset = PartialDataset::new(dataset.clone(), start, end);

            current += batch_size;
            datasets.push(dataset);
        }

        datasets
    }
}

impl<D, I> Dataset<I> for PartialDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = index + self.start_index;
        if index < self.start_index || index >= self.end_index {
            return None;
        }
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        usize::min(self.end_index - self.start_index, self.dataset.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use std::collections::HashSet;

    #[test]
    fn test_start_from_beginning() {
        let dataset_original = FakeDataset::<String>::new(27);
        let mut items_original_1 = HashSet::new();
        let mut items_original_2 = HashSet::new();
        let mut items_partial = HashSet::new();
        dataset_original.iter().enumerate().for_each(|(i, item)| {
            match i >= 10 {
                true => items_original_2.insert(item),
                false => items_original_1.insert(item),
            };
        });

        let dataset_partial = PartialDataset::new(dataset_original, 0, 10);

        for item in dataset_partial.iter() {
            items_partial.insert(item);
        }

        assert_eq!(dataset_partial.len(), 10);
        assert_eq!(items_original_1, items_partial);
        for item in items_original_2 {
            assert!(!items_partial.contains(&item));
        }
    }

    #[test]
    fn test_start_inside() {
        let dataset_original = FakeDataset::<String>::new(27);
        let mut items_original_1 = HashSet::new();
        let mut items_original_2 = HashSet::new();
        let mut items_partial = HashSet::new();

        dataset_original.iter().enumerate().for_each(|(i, item)| {
            match !(10..20).contains(&i) {
                true => items_original_2.insert(item),
                false => items_original_1.insert(item),
            };
        });

        let dataset_partial = PartialDataset::new(dataset_original, 10, 20);
        for item in dataset_partial.iter() {
            items_partial.insert(item);
        }

        assert_eq!(dataset_partial.len(), 10);
        assert_eq!(items_original_1, items_partial);
        for item in items_original_2 {
            assert!(!items_partial.contains(&item));
        }
    }

    #[test]
    fn test_split_contains_all_items_without_duplicates() {
        let dataset_original = FakeDataset::<String>::new(27);
        let mut items_original = Vec::new();
        let mut items_partial = Vec::new();
        for item in dataset_original.iter() {
            items_original.push(item);
        }

        let dataset_partials = PartialDataset::split(dataset_original, 4);

        for dataset in dataset_partials {
            for item in dataset.iter() {
                items_partial.push(item);
            }
        }

        assert_eq!(items_original, items_partial);
    }
}
