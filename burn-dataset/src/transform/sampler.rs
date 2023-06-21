use crate::Dataset;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

/// Sample items from a dataset with replacement.
///
/// This is an efficient way of modeling a dataset as a probability distribution of a fixed size.
pub struct SamplerDataset<D, I> {
    dataset: D,
    size: usize,
    rng: Mutex<StdRng>,
    input: PhantomData<I>,
}

impl<D, I> SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    /// Creates a new sampler dataset.
    pub fn new(dataset: D, size: usize) -> Self {
        let rng = Mutex::new(StdRng::from_entropy());

        Self {
            dataset,
            size,
            rng,
            input: PhantomData,
        }
    }

    /// Generates random index using uniform distribution (0, dataset.len()).
    fn index(&self) -> usize {
        let mut rng = self.rng.lock().unwrap();
        rng.sample(Uniform::new(0, self.dataset.len()))
    }
}

impl<D, I> Dataset<I> for SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    fn get(&self, _index: usize) -> Option<I> {
        self.dataset.get(self.index())
    }

    fn len(&self) -> usize {
        self.size
    }
}
