use super::{
    BatchStrategy, DataLoader, DataLoaderIterator, Progress, TotalCount, batcher::Batcher,
};
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

    fn output_items(&self) -> TotalCount {
        self.strategy.output_items(self.dataset.len())
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
            progress: Progress::new(0, strategy.output_items(len), Some("items".to_string())),
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
    use crate::data::dataloader::{FixBatchStrategy, MultiThreadDataLoader, TotalCount};
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

    /// Reads the entire input before yielding batches, optionally discarding the tail.
    struct BufferedStrategy {
        items: Vec<usize>,
        drop_last: bool,
    }

    impl BatchStrategy<usize> for BufferedStrategy {
        fn output_items(&self, input_items: usize) -> TotalCount {
            TotalCount::Exact(if self.drop_last {
                input_items / 2 * 2
            } else {
                input_items
            })
        }

        fn add(&mut self, item: usize) {
            self.items.push(item);
        }

        fn batch(&mut self, force: bool) -> Option<Vec<usize>> {
            if !force || self.items.is_empty() {
                return None;
            }
            if self.drop_last && self.items.len() < 2 {
                self.items.clear();
                return None;
            }
            Some(self.items.drain(..self.items.len().min(2)).collect())
        }

        fn clone_dyn(&self) -> Box<dyn BatchStrategy<usize>> {
            Box::new(Self {
                items: Vec::new(),
                drop_last: self.drop_last,
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

    #[rstest::rstest]
    #[case::single_keep_tail(0, false)]
    #[case::single_drop_tail(0, true)]
    #[case::threaded_keep_tail(3, false)]
    #[case::threaded_drop_tail(3, true)]
    fn progress_counts_only_yielded_items(#[case] workers: usize, #[case] drop_last: bool) {
        let loader = loader(
            Box::new(BufferedStrategy {
                items: Vec::new(),
                drop_last,
            }),
            11,
            workers,
        );
        let mut iterator = loader.iter();
        assert_eq!(iterator.progress().items_processed, 0);
        let mut emitted = 0;
        while let Some(batch) = iterator.next() {
            emitted += batch.unwrap().len();
            let progress = iterator.progress();
            assert_eq!(progress.items_processed, emitted);
            assert_eq!(progress.total, loader.output_items());
        }
        // Completing with fewer items must not fabricate progress for the dropped tail.
        assert_eq!(emitted, if drop_last { 10 } else { 11 });
        assert_eq!(iterator.progress().items_processed, emitted);
        assert_eq!(iterator.progress().total, loader.output_items());
    }

    #[rstest::rstest]
    #[case::single(0)]
    #[case::threaded(3)]
    fn empty_loaders_report_zero_progress(#[case] workers: usize) {
        let loader = loader(Box::new(FixBatchStrategy::new(2)), 0, workers);
        let mut iterator = loader.iter();
        assert!(iterator.next().is_none());
        assert_eq!(iterator.progress().items_processed, 0);
        assert_eq!(iterator.progress().total, TotalCount::Exact(0));
    }

    struct Filtered {
        items: Vec<usize>,
        total: TotalCount,
    }

    impl BatchStrategy<usize> for Filtered {
        fn add(&mut self, item: usize) {
            if item.is_multiple_of(2) {
                self.items.push(item);
            }
        }
        fn batch(&mut self, force: bool) -> Option<Vec<usize>> {
            if force && !self.items.is_empty() {
                Some(self.items.drain(..self.items.len().min(2)).collect())
            } else {
                None
            }
        }
        fn batch_size(&self) -> Option<usize> {
            None
        }
        fn output_items(&self, _: usize) -> TotalCount {
            self.total
        }
        fn clone_dyn(&self) -> Box<dyn BatchStrategy<usize>> {
            Box::new(Self {
                items: vec![],
                total: self.total,
            })
        }
    }

    #[rstest::rstest]
    #[case::single_unknown(0, TotalCount::Unknown)]
    #[case::single_estimated(0, TotalCount::Estimated(1))]
    #[case::threaded_unknown(3, TotalCount::Unknown)]
    #[case::threaded_estimated(3, TotalCount::Estimated(1))]
    fn filtering_finishes_with_actual_count_for_uncertain_totals(
        #[case] workers: usize,
        #[case] total: TotalCount,
    ) {
        let loader = loader(
            Box::new(Filtered {
                items: vec![],
                total,
            }),
            11,
            workers,
        );
        let mut iterator = loader.iter();
        let mut emitted = 0;
        while let Some(batch) = iterator.next() {
            emitted += batch.unwrap().len();
            assert_eq!(iterator.progress().items_processed, emitted);
        }
        assert_eq!(emitted, 6);
        assert_eq!(iterator.progress().total, TotalCount::Exact(6));
        assert!(iterator.next().is_none());
    }
    #[test]
    fn transient_read_error_can_be_retried_without_advancing_progress() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct RetryDataset(AtomicBool);
        impl Dataset<usize> for RetryDataset {
            fn get(&self, index: usize) -> Result<usize, burn_dataset::DatasetError> {
                if self.0.swap(false, Ordering::SeqCst) {
                    Err(burn_dataset::DatasetError::new(std::io::Error::other(
                        "retry",
                    )))
                } else {
                    Ok(index)
                }
            }
            fn len(&self) -> usize {
                1
            }
        }
        let loader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(1)),
            Arc::new(RetryDataset(AtomicBool::new(true))),
            Arc::new(TestBatcher),
            Default::default(),
            None,
        );
        let mut iterator = loader.iter();
        assert!(iterator.next().unwrap().is_err());
        assert_eq!(iterator.progress().items_processed, 0);
        assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
        assert_eq!(iterator.progress().items_processed, 1);
        assert!(iterator.next().is_none());
    }
}
