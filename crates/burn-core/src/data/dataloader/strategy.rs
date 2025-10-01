/// A strategy to batch items.
pub trait BatchStrategy<I>: Send + Sync {
    /// Adds an item to the strategy.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to add.
    fn add(&mut self, item: I);

    /// Batches the items.
    ///
    /// # Arguments
    ///
    /// * `force` - Whether to force batching.
    ///
    /// # Returns
    ///
    /// The batched items.
    fn batch(&mut self, force: bool) -> Option<Vec<I>>;

    /// Creates a new strategy of the same type.
    ///
    /// # Returns
    ///
    /// The new strategy.
    fn clone_dyn(&self) -> Box<dyn BatchStrategy<I>>;

    /// Returns the expected batch size for this strategy.
    ///
    /// # Returns
    ///
    /// The batch size, or None if the strategy doesn't have a fixed batch size.
    fn batch_size(&self) -> Option<usize>;
}

/// A strategy to batch items with a fixed batch size.
pub struct FixBatchStrategy<I> {
    items: Vec<I>,
    batch_size: usize,
}

impl<I> FixBatchStrategy<I> {
    /// Creates a new strategy to batch items with a fixed batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size.
    ///
    /// # Returns
    ///
    /// The strategy.
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

        if items.is_empty() {
            return None;
        }

        Some(items)
    }

    fn clone_dyn(&self) -> Box<dyn BatchStrategy<I>> {
        Box::new(Self::new(self.batch_size))
    }

    fn batch_size(&self) -> Option<usize> {
        Some(self.batch_size)
    }
}
