use burn_core::data::dataloader::{DataLoader, DataLoaderIterator, Progress};

pub struct EmptyDataLoader {}

impl<O: 'static> DataLoader<O> for EmptyDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(EmptyIterator::new())
    }

    fn num_items(&self) -> usize {
        0
    }
}

pub struct EmptyIterator<O> {
    phantom: std::marker::PhantomData<O>,
}

impl<O> EmptyIterator<O> {
    fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<O> Iterator for EmptyIterator<O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl<O> DataLoaderIterator<O> for EmptyIterator<O> {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: 0,
            items_total: 0,
        }
    }
}
