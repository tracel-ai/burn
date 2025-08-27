use crate::Dataset;
use std::{marker::PhantomData, sync::Arc};

/// Only use a fraction of an existing dataset lazily.
#[derive(new, Clone)]
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

    /// Splits a dataset by distributing complete chunks/batches across multiple partial datasets.
    pub fn split_chunks(
        dataset: D,
        num: usize,
        batch_size: usize,
    ) -> Vec<PartialDataset<Arc<D>, I>> {
        let dataset = Arc::new(dataset); // cheap cloning.
        let total_items = dataset.len();

        // Total number of complete batches
        let total_batches = total_items.div_ceil(batch_size);
        let batches_per_split = total_batches / num;
        let extra_batches = total_batches % num;

        let mut datasets = Vec::with_capacity(num);
        let mut current_batch = 0;

        for i in 0..num {
            // Extra batches distributed across first splits
            let split_batches = if i < extra_batches {
                batches_per_split + 1
            } else {
                batches_per_split
            };

            let start_batch = current_batch;
            let end_batch = start_batch + split_batches;

            let start_index = start_batch * batch_size;
            let end_index = core::cmp::min(end_batch * batch_size, total_items);

            if start_index < total_items {
                datasets.push(PartialDataset::new(dataset.clone(), start_index, end_index));
            }

            current_batch = end_batch;
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
        let expected_len = [6, 6, 6, 9];

        for (i, dataset) in dataset_partials.iter().enumerate() {
            assert_eq!(dataset.len(), expected_len[i]);
            for item in dataset.iter() {
                items_partial.push(item);
            }
        }

        assert_eq!(items_original, items_partial);
    }

    #[test]
    fn test_split_chunks_contains_all_items_without_duplicates() {
        let dataset_original = FakeDataset::<String>::new(27);
        let mut items_original = Vec::new();
        let mut items_partial = Vec::new();
        for item in dataset_original.iter() {
            items_original.push(item);
        }

        let dataset_partials = PartialDataset::split_chunks(dataset_original, 4, 5);
        // [(2 * 5), (2 * 5), 5, 2] -> 5 complete chunks + 1 incomplete with 2 remaining items
        // OTOH, `split(dataset, 4)` would yield [6, 6, 6, 9] -> 4 incomplete chunks + 4 incomplete with [1, 1, 1, 4]
        let expected_len = [10, 10, 5, 2];

        for (i, dataset) in dataset_partials.iter().enumerate() {
            assert_eq!(dataset.len(), expected_len[i]);
            for item in dataset.iter() {
                items_partial.push(item);
            }
        }

        assert_eq!(items_original, items_partial);
    }
}
