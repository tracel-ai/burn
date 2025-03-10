use super::{
    batcher::DynBatcher, BatchStrategy, DataLoader, DataLoaderIterator, DistributionStrategy,
    DynDataLoader, MultiThreadDataLoader, Progress, State,
};
use burn_dataset::{
    transform::{PartialDataset, ShuffledDataset},
    Dataset,
};
use burn_tensor::backend::Backend;
use rand::{distr::StandardUniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

/// A data loader that can be used to iterate over a dataset in batches.
pub struct BatchDataLoader<B: Backend, I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Box<dyn DynBatcher<B, I, O>>,
    distributor: Box<dyn DistributionStrategy<Resource = B::Device>>,
    rng: Option<Arc<spin::Mutex<rand::rngs::StdRng>>>,
}

impl<B: Backend, I, O> Clone for BatchDataLoader<B, I, O> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone_dyn(),
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone_dyn(),
            distributor: self.distributor.clone_dyn(),
            rng: self.rng.clone(),
        }
    }
}

impl<B: Backend, I, O> BatchDataLoader<B, I, O> {
    /// Creates a new batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `distributor` - The resource distribution strategy.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///               iterator is created.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Box<dyn DynBatcher<B, I, O>>,
        distributor: Box<dyn DistributionStrategy<Resource = B::Device>>,
        rng: Option<rand::rngs::StdRng>,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            distributor,
            rng: rng.map(|rng| Arc::new(spin::Mutex::new(rng))),
        }
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchDataloaderIterator<B: Backend, I, O> {
    current_index: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Box<dyn DynBatcher<B, I, O>>,
    distributor: Box<dyn DistributionStrategy<Resource = B::Device>>,
}

impl<B: Backend, I, O> BatchDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Clone + 'static,
{
    /// Creates a new multi-threaded batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `distributor` - The main resource distribution strategy for data loader collection.
    /// * `distributors` - The resource distribution strategy for each data loader thread.
    ///                    Defaults to the main `distributor` strategy otherwise.
    /// * `num_threads` - The number of threads.
    ///
    /// # Returns
    ///
    /// The multi-threaded batch data loader.
    pub fn multi_thread(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Box<dyn DynBatcher<B, I, O>>,
        num_threads: usize,
        distributor: Box<dyn DistributionStrategy<Resource = B::Device>>,
        distributors: Option<Vec<Box<dyn DistributionStrategy<Resource = B::Device>>>>,
        mut rng: Option<rand::rngs::StdRng>,
    ) -> MultiThreadDataLoader<B, O> {
        let distributors = if let Some(distributors) = distributors {
            assert_eq!(
                distributors.len(),
                num_threads,
                "Should specify one distribution strategy per data loader thread"
            );
            distributors
        } else {
            (0..num_threads).map(|_| distributor.clone_dyn()).collect()
        };

        let datasets = PartialDataset::split(dataset, num_threads);

        let mut dataloaders = Vec::with_capacity(num_threads);

        // Create more rngs from the first one, one for each new dataloader.
        let rngs = (0..num_threads).map(|_| {
            rng.as_mut()
                .map(|rng| StdRng::seed_from_u64(Distribution::sample(&StandardUniform, rng)))
        });

        for ((dataset, rng), distributor) in datasets.into_iter().zip(rngs).zip(distributors) {
            let strategy = strategy.clone_dyn();
            let dataloader = BatchDataLoader::new(
                strategy,
                Arc::new(dataset),
                batcher.clone_dyn(),
                distributor,
                rng,
            );
            let dataloader: Box<dyn DynDataLoader<_>> = Box::new(dataloader);
            dataloaders.push(dataloader);
        }
        MultiThreadDataLoader::new(dataloaders, distributor)
    }
}

impl<B, I, O> DataLoader<O> for BatchDataLoader<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // When starting a new iteration, we first check if the dataloader was created with an rng,
        // implying that we should shuffle the dataset beforehand, while advancing the current
        // rng to ensure that each new iteration shuffles the dataset differently.
        let dataset = match &self.rng {
            Some(rng) => {
                let mut rng = rng.lock();

                Arc::new(ShuffledDataset::with_seed(
                    self.dataset.clone(),
                    rng.sample(StandardUniform),
                ))
            }
            None => self.dataset.clone(),
        };
        Box::new(BatchDataloaderIterator::new(
            self.strategy.clone_dyn(),
            dataset,
            self.batcher.clone_dyn(),
            self.distributor.clone_dyn(),
        ))
    }

    fn num_items(&self) -> usize {
        self.dataset.len()
    }
}

impl<B: Backend, I, O> BatchDataloaderIterator<B, I, O> {
    /// Creates a new batch data loader iterator.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `distributor` - The resource distribution strategy.
    ///
    /// # Returns
    ///
    /// The batch data loader iterator.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Box<dyn DynBatcher<B, I, O>>,
        distributor: Box<dyn DistributionStrategy<Resource = B::Device>>,
    ) -> Self {
        BatchDataloaderIterator {
            current_index: 0,
            strategy,
            dataset,
            batcher,
            distributor,
        }
    }
}

impl<B: Backend, I, O> Iterator for BatchDataloaderIterator<B, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        while let Some(item) = self.dataset.get(self.current_index) {
            self.current_index += 1;
            self.strategy.add(item);

            if let Some(items) = self.strategy.batch(false) {
                let device = self.distributor.next().clone();
                self.distributor.select();
                return Some(self.batcher.batch(items, &device));
            }
        }

        if let Some(items) = self.strategy.batch(true) {
            let device = self.distributor.next().clone();
            self.distributor.select();
            return Some(self.batcher.batch(items, &device));
        }

        None
    }
}

impl<B: Backend, I, O> DataLoaderIterator<O> for BatchDataloaderIterator<B, I, O> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_index, self.dataset.len())
    }

    fn state(&self) -> State {
        let progress = self.progress();
        let resource_id = self.distributor.prev();
        State {
            progress,
            resource_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataloader::FixBatchStrategy;
    use crate::data::dataloader::FixedDistributor;
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_batch_dataloader() {
        let batcher = Box::new(TestBatcher::new());
        let distributor = Box::new(FixedDistributor::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            distributor,
            None,
        );

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
        let batcher = Box::new(TestBatcher::new());
        let distributor = Box::new(FixedDistributor::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader_single_thread = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher.clone_dyn(),
            distributor.clone_dyn(),
            None,
        );
        let dataloader_multi_thread = BatchDataLoader::multi_thread(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            4,
            distributor,
            None,
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
