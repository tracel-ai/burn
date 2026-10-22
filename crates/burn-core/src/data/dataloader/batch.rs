use super::{BatchStrategy, DataLoader, DataLoaderIterator, Progress, batcher::Batcher};
use burn_dataset::{
    Dataset,
    transform::{PartialDataset, ShuffledDataset},
};
use burn_tensor::Device;
use rand::SeedableRng;
use std::ops::DerefMut;
use std::sync::Arc;

/// A data loader that can be used to iterate over a dataset in batches.
pub struct BatchDataLoader<I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
    device: Device,
    rng: Option<Arc<spin::Mutex<rand::rngs::StdRng>>>,
}

impl<I, O> Clone for BatchDataLoader<I, O> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone_dyn(),
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone(),
            device: self.device.clone(),
            rng: self.rng.clone(),
        }
    }
}

impl<I, O> BatchDataLoader<I, O> {
    /// Creates a new batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `device`  - The device to use when loading a batch.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///   iterator is created.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
        device: Device,
        rng: Option<rand::rngs::StdRng>,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            device,
            rng: rng.map(|rng| Arc::new(spin::Mutex::new(rng))),
        }
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchDataloaderIterator<I, O> {
    current_index: usize,
    exhausted: bool,
    progress: Progress,
    len: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
    device: Device,
}

impl<I, O> DataLoader<O> for BatchDataLoader<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // When starting a new iteration, we first check if the dataloader was created with an rng,
        // implying that we should shuffle the dataset beforehand, while advancing the current
        // rng to ensure that each new iteration shuffles the dataset differently.
        let dataset = match &self.rng {
            Some(rng) => Arc::new(ShuffledDataset::new(
                self.dataset.clone(),
                rng.lock().deref_mut(),
            )),
            None => self.dataset.clone(),
        };
        Box::new(BatchDataloaderIterator::new(
            self.strategy.clone_dyn(),
            dataset,
            self.batcher.clone(),
            self.device.clone(),
        ))
    }

    fn expected_num_items(&self) -> Option<usize> {
        self.strategy.expected_num_items(self.dataset.len())
    }

    fn num_items(&self) -> usize {
        self.dataset.len()
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<O>> {
        let rng = self.rng.as_ref().map(|rng| {
            let mut rng = rng.lock();
            rng.fork()
        });
        Arc::new(Self::new(
            self.strategy.clone_dyn(),
            self.dataset.clone(),
            self.batcher.clone(),
            device.clone(),
            rng,
        ))
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<O>> {
        let rng = self.rng.as_ref().map(|rng| {
            let mut rng = rng.lock();
            rng.fork()
        });
        let dataloader = Self::new(
            self.strategy.clone_dyn(),
            Arc::new(PartialDataset::new(self.dataset.clone(), start, end)),
            self.batcher.clone(),
            self.device.clone(),
            rng,
        );
        Arc::new(dataloader)
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
    /// * `device`  - The device to use when loading a batch.
    ///
    /// # Returns
    ///
    /// The batch data loader iterator.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
        device: Device,
    ) -> Self {
        let len = dataset.len();

        BatchDataloaderIterator {
            current_index: 0,
            exhausted: false,
            progress: Progress::new(
                0,
                strategy.expected_num_items(len),
                Some("items".to_string()),
            ),
            len,
            strategy,
            dataset,
            batcher,
            device,
        }
    }
}

impl<I, O> Iterator for BatchDataloaderIterator<I, O> {
    type Item = Result<O, burn_dataset::DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        while self.current_index < self.len {
            let chunk_size = self
                .strategy
                .batch_size()
                .unwrap_or(1)
                .min(self.len - self.current_index);
            let indexes = (self.current_index..self.current_index + chunk_size).collect();

            let items = match self.dataset.get_many(indexes) {
                Ok(items) => items,
                Err(err) => return Some(Err(err)),
            };
            self.current_index += chunk_size;

            for item in items {
                self.strategy.add(item);
            }

            if let Some(items) = self.strategy.batch(false) {
                self.progress.items_processed += items.len();
                return Some(Ok(self.batcher.batch(items, &self.device)));
            }
        }

        if let Some(items) = self.strategy.batch(true) {
            self.progress.items_processed += items.len();
            return Some(Ok(self.batcher.batch(items, &self.device)));
        }

        self.exhausted = true;
        self.progress.resolve_total();
        None
    }
}

impl<I, O> DataLoaderIterator<O> for BatchDataloaderIterator<I, O> {
    fn progress(&self) -> Progress {
        self.progress.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataloader::{FixBatchStrategy, MultiThreadDataLoader};
    use crate::data::dataset::{FakeDataset, InMemDataset};

    #[test]
    fn test_batch_dataloader() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            Default::default(),
            None,
        );

        let mut items_dataset = HashSet::new();
        let mut items_dataloader = HashSet::new();

        for item in dataset.iter().map(Result::unwrap) {
            items_dataset.insert(item);
        }

        for items in dataloader.iter().map(Result::unwrap) {
            for item in items {
                items_dataloader.insert(item);
            }
        }

        assert_eq!(items_dataset, items_dataloader);
    }

    #[test]
    fn test_batch_dataloader_slice() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            Default::default(),
            None,
        );
        let dataloader_slice = dataloader.slice(5, 15);

        let mut items_dataloader = HashSet::new();
        let mut items_dataloader_slice = HashSet::new();

        let mut idx = 0;
        for items in dataloader.iter().map(Result::unwrap) {
            for item in items {
                if (5..15).contains(&idx) {
                    items_dataloader.insert(item);
                }
                idx += 1;
            }
        }

        for items in dataloader_slice.iter().map(Result::unwrap) {
            for item in items {
                items_dataloader_slice.insert(item);
            }
        }

        assert_eq!(items_dataloader, items_dataloader_slice);
    }

    #[test]
    fn test_batch_dataloader_incomplete_last_batch() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            Default::default(),
            None,
        );

        let batch_sizes: Vec<usize> = dataloader
            .iter()
            .map(Result::unwrap)
            .map(|items| items.len())
            .collect();

        assert_eq!(batch_sizes, vec![5, 5, 5, 5, 5, 2]);
    }

    #[test]
    fn test_batch_dataloader_exact_multiple_batches() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(25));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            Default::default(),
            None,
        );

        let batch_sizes: Vec<usize> = dataloader
            .iter()
            .map(Result::unwrap)
            .map(|items| items.len())
            .collect();

        assert_eq!(batch_sizes, vec![5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_batch_dataloader_empty_dataset() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(0));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            Default::default(),
            None,
        );

        assert!(dataloader.iter().next().is_none());
    }

    /// Buffers the whole input and drops an incomplete pair.
    struct BufferedPairs {
        items: Vec<usize>,
        known_total: bool,
    }

    impl BatchStrategy<usize> for BufferedPairs {
        fn expected_num_items(&self, input_items: usize) -> Option<usize> {
            if self.known_total {
                Some(input_items / 2 * 2)
            } else {
                None
            }
        }
        fn add(&mut self, item: usize) {
            self.items.push(item);
        }
        fn batch(&mut self, force: bool) -> Option<Vec<usize>> {
            (force && self.items.len() >= 2).then(|| self.items.drain(..2).collect())
        }
        fn clone_dyn(&self) -> Box<dyn BatchStrategy<usize>> {
            Box::new(Self {
                items: vec![],
                known_total: self.known_total,
            })
        }
        fn batch_size(&self) -> Option<usize> {
            Some(2)
        }
    }

    fn loader(
        strategy: Box<dyn BatchStrategy<usize>>,
        len: usize,
        workers: usize,
    ) -> Box<dyn DataLoader<Vec<usize>>> {
        let dataset = Arc::new(InMemDataset::new((0..len).collect()));
        let batcher = Arc::new(TestBatcher::new());
        if workers == 0 {
            Box::new(BatchDataLoader::new(
                strategy,
                dataset,
                batcher,
                Default::default(),
                None,
            ))
        } else {
            Box::new(MultiThreadDataLoader::new(
                strategy,
                dataset,
                batcher,
                workers,
                Default::default(),
                None,
            ))
        }
    }

    #[test]
    fn buffered_progress_counts_emitted_items_and_resolves_unknown_totals() {
        for workers in [0, 3] {
            for known_total in [false, true] {
                let loader = loader(
                    Box::new(BufferedPairs {
                        items: vec![],
                        known_total,
                    }),
                    11,
                    workers,
                );
                let mut iterator = loader.iter();
                let initial_total = if known_total { Some(10) } else { None };
                assert_eq!(loader.expected_num_items(), initial_total);
                assert_eq!(iterator.progress().total, initial_total);
                assert_eq!(iterator.progress().items_processed, 0);
                let mut emitted = 0;
                while let Some(batch) = iterator.next() {
                    emitted += batch.unwrap().len();
                    assert_eq!(iterator.progress().items_processed, emitted);
                }
                assert_eq!(emitted, 10);
                assert_eq!(iterator.progress().items_processed, 10);
                assert_eq!(iterator.progress().total, Some(10));
                assert!(iterator.next().is_none());
            }
        }
    }

    #[test]
    fn empty_loaders_report_zero_progress() {
        for workers in [0, 3] {
            let loader = loader(Box::new(FixBatchStrategy::new(2)), 0, workers);
            let mut iterator = loader.iter();
            assert!(iterator.next().is_none());
            assert_eq!(iterator.progress().items_processed, 0);
            assert_eq!(iterator.progress().total, Some(0));
        }
    }
}
