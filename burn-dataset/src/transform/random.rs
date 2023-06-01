use crate::Dataset;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::marker::PhantomData;

/// Shuffled a dataset, consider using [sampler dataset](crate::transform::SamplerDataset) is you
/// want a probability distribution that is computed lazily.
pub struct ShuffledDataset<D, I> {
    dataset: D,
    indexes: Vec<usize>,
    input: PhantomData<I>,
}

impl<D, I> ShuffledDataset<D, I>
where
    D: Dataset<I>,
{
    pub fn new(dataset: D, rng: &mut StdRng) -> Self {
        let mut indexes = Vec::with_capacity(dataset.len());
        for i in 0..dataset.len() {
            indexes.push(i);
        }
        indexes.shuffle(rng);

        Self {
            dataset,
            indexes,
            input: PhantomData,
        }
    }

    pub fn with_seed(dataset: D, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(dataset, &mut rng)
    }
}

impl<D, I> Dataset<I> for ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = match self.indexes.get(index) {
            Some(index) => index,
            None => return None,
        };
        self.dataset.get(*index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
