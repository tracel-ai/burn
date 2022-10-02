use crate::Dataset;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use std::sync::Mutex;

pub struct SamplerDataset<I> {
    dataset: Box<dyn Dataset<I>>,
    size: usize,
    rng: Mutex<StdRng>,
}

impl<I> SamplerDataset<I> {
    pub fn from_dataset<D: Dataset<I> + 'static>(dataset: D, size: usize) -> Self {
        Self::new(Box::new(dataset), size)
    }

    pub fn new(dataset: Box<dyn Dataset<I>>, size: usize) -> Self {
        let rng = Mutex::new(StdRng::from_entropy());

        Self { dataset, size, rng }
    }

    fn index(&self) -> usize {
        let distribution = Uniform::new(0, self.dataset.len());
        let mut rng = self.rng.lock().unwrap();
        rng.sample(distribution)
    }
}

impl<I> Dataset<I> for SamplerDataset<I> {
    fn get(&self, _index: usize) -> Option<I> {
        self.dataset.get(self.index())
    }

    fn len(&self) -> usize {
        self.size
    }
}
