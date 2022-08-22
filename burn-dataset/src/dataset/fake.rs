use crate::{Dataset, DatasetIterator, InMemDataset};
use fake::{Dummy, Fake, Faker};

pub struct FakeDataset<I> {
    dataset: InMemDataset<I>,
}

impl<I: Dummy<Faker>> FakeDataset<I> {
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
    fn iter<'a>(&'a self) -> DatasetIterator<'a, I> {
        DatasetIterator::new(self)
    }

    fn get(&self, index: usize) -> Option<I> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
