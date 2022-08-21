use super::{batcher::Batcher, DataLoader};
use burn_dataset::{Dataset, DatasetIterator};

pub struct BasicDataLoader<I, O> {
    batch_size: usize,
    dataset: Box<dyn Dataset<I>>,
    batcher: Box<dyn Batcher<I, O>>,
}

struct BasicDataloaderIterator<'a, I, O> {
    batch_size: usize,
    dataset: DatasetIterator<'a, I>,
    batcher: &'a Box<dyn Batcher<I, O>>,
}

impl<I, O> BasicDataLoader<I, O> {
    pub fn new(
        batch_size: usize,
        dataset: Box<dyn Dataset<I>>,
        batcher: Box<dyn Batcher<I, O>>,
    ) -> Self {
        Self {
            batch_size,
            dataset,
            batcher,
        }
    }
}

impl<I, O> DataLoader<O> for BasicDataLoader<I, O> {
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = O> + 'a> {
        Box::new(BasicDataloaderIterator::new(
            self.batch_size,
            &self.dataset,
            &self.batcher,
        ))
    }

    fn len(&self) -> usize {
        self.dataset.len() / self.batch_size
    }
}

impl<'a, I, O> BasicDataloaderIterator<'a, I, O> {
    pub fn new(
        batch_size: usize,
        dataset: &'a Box<dyn Dataset<I>>,
        batcher: &'a Box<dyn Batcher<I, O>>,
    ) -> Self {
        BasicDataloaderIterator {
            batch_size,
            dataset: dataset.iter(),
            batcher,
        }
    }
}

impl<'a, I, O> Iterator for BasicDataloaderIterator<'a, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        let mut items = Vec::with_capacity(self.batch_size);
        loop {
            if items.len() >= self.batch_size {
                break;
            }

            let item = self.dataset.next();
            let item = match item {
                Some(item) => item,
                None => break,
            };
            items.push(item);
        }
        if items.len() == 0 {
            return None;
        }

        let batch = self.batcher.batch(items);
        Some(batch)
    }
}
