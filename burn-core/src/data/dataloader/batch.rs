use super::{
    batcher::Batcher, BatchStrategy, DataLoader, DataLoaderIterator, MultiThreadDataLoader,
    Progress,
};
use burn_dataset::{transform::PartialDataset, Dataset};
use std::sync::Arc;

/// A data loader that can be used to iterate over a dataset in batches.
pub struct BatchDataLoader<I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchDataloaderIterator<I, O> {
    current_index: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
}

impl<I, O> BatchDataLoader<I, O> {
    /// Creates a new batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
        }
    }
}

impl<I, O> BatchDataLoader<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Sync + Clone + 'static,
{
    /// Creates a new multi-threaded batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `num_threads` - The number of threads.
    ///
    /// # Returns
    ///
    /// The multi-threaded batch data loader.
    pub fn multi_thread(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
        num_threads: usize,
    ) -> MultiThreadDataLoader<O> {
        let datasets = PartialDataset::split(dataset, num_threads);
        let mut dataloaders: Vec<Arc<dyn DataLoader<_> + Send + Sync>> = Vec::new();
        for dataset in datasets {
            let strategy = strategy.new_like();
            let dataloader = BatchDataLoader::new(strategy, Arc::new(dataset), batcher.clone());
            let dataloader = Arc::new(dataloader);
            dataloaders.push(dataloader);
        }
        MultiThreadDataLoader::new(dataloaders)
    }
}

impl<I, O> DataLoader<O> for BatchDataLoader<I, O> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(BatchDataloaderIterator::new(
            self.strategy.new_like(),
            self.dataset.clone(),
            self.batcher.clone(),
        ))
    }
}

impl<I, O> BatchDataloaderIterator<I, O> {
    /// Creates a new batch data loader iterator.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    ///
    /// # Returns
    ///
    /// The batch data loader iterator.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
    ) -> Self {
        BatchDataloaderIterator {
            current_index: 0,
            strategy,
            dataset,
            batcher,
        }
    }
}

impl<I, O> Iterator for BatchDataloaderIterator<I, O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        loop {
            let item = self.dataset.get(self.current_index);
            self.current_index += 1;

            let item = match item {
                Some(item) => item,
                None => break,
            };
            self.strategy.add(item);

            if let Some(items) = self.strategy.batch(false) {
                return Some(self.batcher.batch(items));
            }
        }

        if let Some(items) = self.strategy.batch(true) {
            return Some(self.batcher.batch(items));
        }

        None
    }
}

impl<I, O> DataLoaderIterator<O> for BatchDataloaderIterator<I, O> {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.current_index,
            items_total: self.dataset.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataloader::FixBatchStrategy;
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_batch_dataloader() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader =
            BatchDataLoader::new(Box::new(FixBatchStrategy::new(5)), dataset.clone(), batcher);

        let mut items_dataset = HashSet::new();
        let mut items_dataloader = HashSet::new();

        for item in dataset.iter() {
            items_dataset.insert(item);
        }

        for items in dataloader.iter() {
            for item in items {
                items_dataloader.insert(item);
            }
        }

        assert_eq!(items_dataset, items_dataloader);
    }

    #[test]
    fn test_multi_thread_batch_dataloader() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader_single_thread = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher.clone(),
        );
        let dataloader_multi_thread =
            BatchDataLoader::multi_thread(Box::new(FixBatchStrategy::new(5)), dataset, batcher, 4);

        let mut items_single_thread = HashSet::new();
        let mut items_multi_thread = HashSet::new();

        for items in dataloader_single_thread.iter() {
            for item in items {
                items_single_thread.insert(item);
            }
        }

        for items in dataloader_multi_thread.iter() {
            for item in items {
                items_multi_thread.insert(item);
            }
        }

        assert_eq!(items_single_thread, items_multi_thread);
    }
}
