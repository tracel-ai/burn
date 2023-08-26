use crate::Dataset;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use std::{marker::PhantomData, sync::Mutex};

/// Sample items from a dataset.
///
/// This is an convenient way of modeling a dataset as a probability distribution of a fixed size.
/// You have multiple options to instantiate the dataset sampler.
///
/// * With replacement: This is the most efficient way of using the sampler because no state is
///   required to keep indexes that have been selected.
///
/// * Without replacement: This has a similar effect to using a
///   [shuffled dataset](crate::transform::ShuffledDataset), but with more flexibility since you can
///   set the dataset to an arbitrary size.
pub struct SamplerDataset<D, I> {
    dataset: D,
    size: usize,
    rng: Mutex<StdRng>,
    state: Option<Mutex<Vec<usize>>>,
    input: PhantomData<I>,
}

impl<D, I> SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    /// Creates a new sampler dataset with replacement.
    pub fn new(dataset: D, size: usize) -> Self {
        let rng = Mutex::new(StdRng::from_entropy());

        Self {
            dataset,
            size,
            rng,
            state: None,
            input: PhantomData,
        }
    }

    /// Creates a new sampler dataset with replacement.
    pub fn with_replacement(dataset: D, size: usize) -> Self {
        Self::new(dataset, size)
    }

    /// Creates a new sampler dataset without replacement.
    pub fn without_replacement(dataset: D, size: usize) -> Self {
        let rng = Mutex::new(StdRng::from_entropy());
        let state = Mutex::new((0..dataset.len()).collect());

        Self {
            dataset,
            size,
            rng,
            state: Some(state),
            input: PhantomData,
        }
    }

    fn index(&self) -> usize {
        match &self.state {
            Some(state) => {
                let mut state = state.lock().unwrap();

                if state.len() == 0 {
                    // Refill the state using the same vector.
                    (0..self.dataset.len()).for_each(|i| state.push(i));
                }

                let index = self.index_size(state.len());
                state.remove(index)
            }
            None => self.index_size(self.dataset.len()),
        }
    }

    fn index_size(&self, size: usize) -> usize {
        let mut rng = self.rng.lock().unwrap();
        rng.sample(Uniform::new(0, size))
    }
}

impl<D, I> Dataset<I> for SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        if index >= self.size {
            return None;
        }

        self.dataset.get(self.index())
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use std::collections::HashMap;

    #[test]
    fn sampler_dataset_with_replacement_iter() {
        let factor = 3;
        let len_original = 10;
        let dataset_sampler = SamplerDataset::with_replacement(
            FakeDataset::<String>::new(len_original),
            len_original * factor,
        );
        let mut total = 0;

        for _item in dataset_sampler.iter() {
            total += 1;
        }

        assert_eq!(total, factor * len_original);
    }

    #[test]
    fn sampler_dataset_without_replacement_bucket_test() {
        let factor = 3;
        let len_original = 10;
        let dataset_sampler = SamplerDataset::without_replacement(
            FakeDataset::<String>::new(len_original),
            len_original * factor,
        );
        let mut buckets = HashMap::new();

        for item in dataset_sampler.iter() {
            let count = match buckets.get(&item) {
                Some(count) => count + 1,
                None => 1,
            };

            buckets.insert(item, count);
        }

        let mut total = 0;
        for count in buckets.into_values() {
            assert_eq!(count, factor);
            total += count;
        }
        assert_eq!(total, factor * len_original);
    }
}
