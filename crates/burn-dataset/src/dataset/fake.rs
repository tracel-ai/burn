use crate::{Dataset, DatasetIterator, InMemDataset};
use fake::{Dummy, Fake, Faker};

/// Dataset filled with fake items generated from the [fake](fake) crate.
pub struct FakeDataset<I> {
    dataset: InMemDataset<I>,
}

impl<I: Dummy<Faker>> FakeDataset<I> {
    /// Create a new fake dataset with the given size.
    pub fn new(size: usize) -> Self {
        let mut items = Vec::with_capacity(size);
        for _ in 0..size {
            items.push(Faker.fake());
        }
        let dataset = InMemDataset::new(items);

        Self { dataset }
    }
}

impl<I: Send + Sync + Clone> Dataset<I> for FakeDataset<I> {
    fn iter(&self) -> DatasetIterator<'_, I> {
        DatasetIterator::new(self)
    }

    fn get(&self, index: usize) -> Option<I> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}
