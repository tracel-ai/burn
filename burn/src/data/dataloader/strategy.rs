pub trait BatchStrategy<I>: Send + Sync {
    fn add(&mut self, item: I);
    fn batch(&mut self, force: bool) -> Option<Vec<I>>;
    fn new_like(&self) -> Box<dyn BatchStrategy<I>>;
}

pub struct FixBatchStrategy<I> {
    items: Vec<I>,
    batch_size: usize,
}

impl<I> FixBatchStrategy<I> {
    pub fn new(batch_size: usize) -> Self {
        FixBatchStrategy {
            items: Vec::with_capacity(batch_size),
            batch_size,
        }
    }
}

impl<I: Send + Sync + 'static> BatchStrategy<I> for FixBatchStrategy<I> {
    fn add(&mut self, item: I) {
        self.items.push(item);
    }

    fn batch(&mut self, force: bool) -> Option<Vec<I>> {
        if self.items.len() < self.batch_size && !force {
            return None;
        }

        let mut items = Vec::with_capacity(self.batch_size);
        std::mem::swap(&mut items, &mut self.items);

        if items.len() == 0 {
            return None;
        }

        Some(items)
    }

    fn new_like(&self) -> Box<dyn BatchStrategy<I>> {
        Box::new(Self::new(self.batch_size))
    }
}
