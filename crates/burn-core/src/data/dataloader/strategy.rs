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
    drop_last: bool,
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
            drop_last: false,
        }
    }

    /// Sets whether to drop the last batch when it is smaller than the batch size.
    ///
    /// By default, the last incomplete batch is kept.
    ///
    /// # Arguments
    ///
    /// * `drop_last` - Whether to drop the last incomplete batch.
    ///
    /// # Returns
    ///
    /// The strategy.
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
}

impl<I: Send + Sync + 'static> BatchStrategy<I> for FixBatchStrategy<I> {
    fn add(&mut self, item: I) {
        self.items.push(item);
    }

    fn batch(&mut self, force: bool) -> Option<Vec<I>> {
        if self.items.len() < self.batch_size {
            if !force {
                return None;
            }

            if self.drop_last {
                self.items.clear();
                return None;
            }
        }

        let mut items = Vec::with_capacity(self.batch_size);
        std::mem::swap(&mut items, &mut self.items);

        if items.is_empty() {
            return None;
        }

        Some(items)
    }

    fn clone_dyn(&self) -> Box<dyn BatchStrategy<I>> {
        Box::new(Self::new(self.batch_size).with_drop_last(self.drop_last))
    }

    fn batch_size(&self) -> Option<usize> {
        Some(self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_sizes(mut strategy: FixBatchStrategy<usize>, num_items: usize) -> Vec<usize> {
        let mut sizes = Vec::new();

        for item in 0..num_items {
            strategy.add(item);
            if let Some(items) = strategy.batch(false) {
                sizes.push(items.len());
            }
        }
        if let Some(items) = strategy.batch(true) {
            sizes.push(items.len());
        }

        sizes
    }

    #[test]
    fn keeps_the_last_incomplete_batch_by_default() {
        assert_eq!(batch_sizes(FixBatchStrategy::new(4), 10), vec![4, 4, 2]);
    }

    #[test]
    fn drops_the_last_incomplete_batch() {
        let strategy = FixBatchStrategy::new(4).with_drop_last(true);

        assert_eq!(batch_sizes(strategy, 10), vec![4, 4]);
    }

    #[test]
    fn drop_last_keeps_a_complete_last_batch() {
        let strategy = FixBatchStrategy::new(4).with_drop_last(true);

        assert_eq!(batch_sizes(strategy, 8), vec![4, 4]);
    }

    #[test]
    fn drop_last_yields_nothing_when_the_dataset_is_smaller_than_a_batch() {
        let strategy = FixBatchStrategy::new(4).with_drop_last(true);

        assert!(batch_sizes(strategy, 3).is_empty());
    }

    #[test]
    fn drop_last_survives_clone_dyn() {
        let strategy = FixBatchStrategy::<usize>::new(4).with_drop_last(true);
        let mut cloned = strategy.clone_dyn();

        for item in 0..3 {
            cloned.add(item);
        }

        assert!(cloned.batch(true).is_none());
    }
}
