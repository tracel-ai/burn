use burn_dataset::Dataset;
use burn_dataset::transform::PartialDataset;
use burn_tensor::backend::Backend;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;

use super::batcher::DynBatcher;
use super::{BatchDataLoader, BatchStrategy, DataLoader, DataLoaderIterator, Progress};
use core::cell::OnceCell;
use std::sync::{Arc, mpsc};
use std::thread;

const MAX_QUEUED_ITEMS: usize = 100;

/// A multi-threaded data loader that can be used to iterate over a dataset.
pub struct MultiThreadDataLoader<B: Backend, I, O> {
    // Configuration parameters needed for initialization
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Box<dyn DynBatcher<B, I, O>>,
    device: B::Device,
    rng: Option<rand::rngs::StdRng>,
    num_threads: usize,

    // The lazily initialized data loaders
    dataloaders: OnceCell<Vec<BatchDataLoader<B, I, O>>>,
}

/// A message that can be sent between threads.
#[derive(Debug)]
pub enum Message<O> {
    /// A batch of items.
    Batch(usize, O, Progress),

    /// The thread is done.
    Done,
}

struct MultiThreadsDataloaderIterator<O> {
    num_done: usize,
    workers: Vec<thread::JoinHandle<()>>,
    receiver: mpsc::Receiver<Message<O>>,
    progresses: Vec<Progress>,
}

impl<B: Backend, I, O> MultiThreadDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    /// Creates a new multi-threaded batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `num_threads` - The number of threads.
    /// * `device`  - The device to use when loading a batch.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///   iterator is created.
    ///
    /// # Returns
    ///
    /// The multi-threaded batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Box<dyn DynBatcher<B, I, O>>,
        num_threads: usize,
        device: B::Device,
        rng: Option<rand::rngs::StdRng>,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            num_threads,
            device,
            rng,
            dataloaders: OnceCell::new(),
        }
    }

    /// Force initialization if needed.
    fn initialize(&self) -> &[BatchDataLoader<B, I, O>] {
        self.dataloaders
            .get_or_init(|| {
                let datasets = PartialDataset::split(self.dataset.clone(), self.num_threads);

                // Create more rngs from the first one, one for each new dataloader.
                let mut rng = self.rng.clone();
                let rngs = (0..self.num_threads).map(|_| {
                    rng.as_mut().map(|rng| {
                        StdRng::seed_from_u64(Distribution::sample(&StandardUniform, rng))
                    })
                });

                datasets
                    .into_iter()
                    .zip(rngs)
                    .map(|(dataset, rng)| {
                        let strategy = self.strategy.clone_dyn();
                        BatchDataLoader::new(
                            strategy,
                            Arc::new(dataset),
                            self.batcher.clone_dyn(),
                            self.device.clone(),
                            rng,
                        )
                    })
                    .collect()
            })
            .as_ref()
    }
}

impl<B: Backend, I, O> DataLoader<B, O> for MultiThreadDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static + std::fmt::Debug,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // This will initialize the loader if it hasn't been initialized yet
        let dataloaders = self.initialize();

        let (sender, receiver) = mpsc::sync_channel::<Message<O>>(MAX_QUEUED_ITEMS);

        let mut progresses = Vec::with_capacity(dataloaders.len());

        let handlers: Vec<_> = dataloaders
            .iter()
            .enumerate()
            .map(|(index, dataloader)| {
                let dataloader_cloned = dataloader.clone();
                let sender_cloned = sender.clone();
                progresses.push(Progress::new(0, dataloader_cloned.num_items()));

                thread::spawn(move || {
                    let mut iterator = dataloader_cloned.iter();
                    while let Some(item) = iterator.next() {
                        let progress = iterator.progress();

                        match sender_cloned.send(Message::Batch(index, item, progress)) {
                            Ok(_) => {}
                            // The receiver is probably gone, no need to panic, just need to stop
                            // iterating.
                            Err(_) => return,
                        };
                    }
                    // Same thing.
                    sender_cloned.send(Message::Done).ok();
                })
            })
            .collect();

        Box::new(MultiThreadsDataloaderIterator::new(
            receiver, handlers, progresses,
        ))
    }

    fn num_items(&self) -> usize {
        // For num_items, we can directly use the dataset size without
        // necessarily initializing the full loader
        self.dataset.len()
    }

    fn set_device(&mut self, device: B::Device) {
        self.device = device;
    }

    fn slice(&self, start: usize, end: usize) -> Box<dyn DataLoader<B, O>> {
        let dataloader = Self::new(
            self.strategy.clone_dyn(),
            Arc::new(PartialDataset::new(self.dataset.clone(), start, end)),
            self.batcher.clone_dyn(),
            self.num_threads,
            self.device.clone(),
            self.rng.clone(),
        );
        Box::new(dataloader)
    }
}

impl<O> MultiThreadsDataloaderIterator<O> {
    pub fn new(
        receiver: mpsc::Receiver<Message<O>>,
        workers: Vec<thread::JoinHandle<()>>,
        progresses: Vec<Progress>,
    ) -> Self {
        MultiThreadsDataloaderIterator {
            num_done: 0,
            workers,
            receiver,
            progresses,
        }
    }
}
impl<O: std::fmt::Debug> DataLoaderIterator<O> for MultiThreadsDataloaderIterator<O> {
    fn progress(&self) -> Progress {
        let mut items_total = 0;
        let mut items_processed = 0;

        for progress in self.progresses.iter() {
            items_total += progress.items_total;
            items_processed += progress.items_processed;
        }

        Progress::new(items_processed, items_total)
    }
}

impl<O: std::fmt::Debug> Iterator for MultiThreadsDataloaderIterator<O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        if self.workers.is_empty() {
            return None;
        }

        loop {
            let item = self.receiver.recv();
            let item = item.unwrap();

            match item {
                Message::Batch(index, item, progress) => {
                    if let Some(current) = self.progresses.get_mut(index) {
                        *current = progress;
                    }
                    return Some(item);
                }
                Message::Done => {
                    self.num_done += 1;
                }
            };

            if self.num_done == self.workers.len() {
                while let Some(worker) = self.workers.pop() {
                    worker.join().unwrap();
                }
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::FixBatchStrategy;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_multi_thread_batch_dataloader() {
        let batcher = Box::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader_single_thread = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher.clone_dyn(),
            Default::default(),
            None,
        );
        let dataloader_multi_thread = MultiThreadDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            4,
            Default::default(),
            None,
        );

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
