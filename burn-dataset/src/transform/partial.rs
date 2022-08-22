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

impl<I> Dataset<I> for PartialDataset<I>
where
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = index + self.start_index;
        if index < self.start_index || index >= self.end_index {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use std::collections::HashSet;

    #[test]
    fn test_start_from_beginning() {
        let dataset_original = Arc::new(FakeDataset::<String>::new(27));
        let dataset_partial = PartialDataset::new(dataset_original.clone(), 0, 10);

        let mut items_original_1 = HashSet::new();
        let mut items_original_2 = HashSet::new();
        let mut items_partial = HashSet::new();

        for (i, item) in dataset_original.iter().enumerate() {
            if i >= 10 {
                items_original_2.insert(item);
            } else {
                items_original_1.insert(item);
            }
        }

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
        let dataset_original = Arc::new(FakeDataset::<String>::new(27));
        let dataset_partial = PartialDataset::new(dataset_original.clone(), 10, 20);

        let mut items_original_1 = HashSet::new();
        let mut items_original_2 = HashSet::new();
        let mut items_partial = HashSet::new();

        for (i, item) in dataset_original.iter().enumerate() {
            if i < 10 || i >= 20 {
                items_original_2.insert(item);
            } else {
                items_original_1.insert(item);
            }
        }

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
        let dataset_original = Arc::new(FakeDataset::<String>::new(27));
        let dataset_partials = PartialDataset::split(dataset_original.clone(), 4);

        let mut items_original = Vec::new();
        let mut items_partial = Vec::new();

        for item in dataset_original.iter() {
            items_original.push(item);
        }

        for dataset in dataset_partials {
            for item in dataset.iter() {
                items_partial.push(item);
            }
        }

        assert_eq!(items_original, items_partial);
    }
}
