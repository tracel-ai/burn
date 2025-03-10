/// A strategy to batch items.
pub trait BatchStrategy<I>: Send {
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

impl<I: Send + 'static> BatchStrategy<I> for FixBatchStrategy<I> {
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
}

/// A strategy for distributing items across devices or other resources.
pub trait DistributionStrategy: Send {
    /// The resource type.
    type Resource: Send + Clone;

    /// Create a new distribution strategy for the specified resources.
    fn new(resources: Vec<Self::Resource>) -> Self
    where
        Self: Sized;

    /// Predicts the next resource to assign an item to.
    fn next(&mut self) -> &Self::Resource {
        let id = self.next_id();
        &self.resources()[id]
    }

    /// Predicts the next resource to assign an item to.
    fn next_id(&mut self) -> usize;

    /// Returns the strategy resources.
    fn resources(&self) -> &[Self::Resource];

    /// Returns the previous resource id assigned.
    fn prev(&self) -> Option<usize>;

    /// Marks the selection of the predicted resource.
    /// If the strategy is not stateful (i.e., not history dependent), this can be a no-op.
    fn select(&mut self);

    /// Creates a new strategy of the same type.
    fn clone_dyn(&self) -> Box<dyn DistributionStrategy<Resource = Self::Resource>>;
}

/// Always selects the same resource or device.
pub struct FixedDistributor<R> {
    resources: Vec<R>,
    fixed_id: usize,
    selected: bool,
}

impl<R> FixedDistributor<R> {
    /// Sets the fixed resource id.
    pub fn with_fixed(mut self, resource_id: usize) -> Self {
        self.fixed_id = resource_id;
        self
    }
}

impl<R: Send + Clone + 'static> DistributionStrategy for FixedDistributor<R> {
    type Resource = R;

    fn new(resources: Vec<Self::Resource>) -> Self
    where
        Self: Sized,
    {
        Self {
            resources,
            fixed_id: 0,
            selected: false,
        }
    }

    fn next_id(&mut self) -> usize {
        self.fixed_id
    }

    fn prev(&self) -> Option<usize> {
        if self.selected {
            Some(self.fixed_id)
        } else {
            None
        }
    }

    fn select(&mut self) {
        self.selected = true
    }

    fn clone_dyn(&self) -> Box<dyn DistributionStrategy<Resource = R>> {
        Box::new(Self::new(self.resources.clone()).with_fixed(self.fixed_id))
    }

    fn resources(&self) -> &[Self::Resource] {
        self.resources.as_slice()
    }
}

/// A strategy for assigning items to resources in a round-robin fashion.
/// It cycles through the available resources, ensuring that batches are
/// assigned in turn to each resource, balancing the load evenly across them.
#[derive(Clone, Debug)]
pub struct RoundRobinDistributor<R> {
    resources: Vec<R>,
    total: usize,
    next: usize,
    prev: Option<usize>,
}

impl<R> RoundRobinDistributor<R> {
    fn compute_next(&self, value: Option<usize>) -> usize {
        value.map_or(0, |x| (x + 1).wrapping_rem(self.total))
    }
}

impl<R: Send + Clone + 'static> DistributionStrategy for RoundRobinDistributor<R> {
    type Resource = R;

    fn new(resources: Vec<R>) -> Self
    where
        Self: Sized,
    {
        let total = resources.len();
        Self {
            resources,
            total,
            next: 0,
            prev: None,
        }
    }
    fn next_id(&mut self) -> usize {
        self.next = self.compute_next(self.prev);
        self.next
    }

    fn prev(&self) -> Option<usize> {
        self.prev
    }

    fn select(&mut self) {
        if self.next == self.compute_next(self.prev) {
            self.prev = Some(self.next)
        }
    }

    fn clone_dyn(&self) -> Box<dyn DistributionStrategy<Resource = R>> {
        Box::new(Self::new(self.resources.clone()))
    }

    fn resources(&self) -> &[Self::Resource] {
        self.resources.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_device() {
        let device_id = 1;
        let mut distributor = FixedDistributor::new(vec![0, device_id]).with_fixed(device_id);

        // Nothing selected yet
        assert_eq!(distributor.prev(), None);

        // Predict next, but not marked as selected/used
        assert_eq!(*distributor.next(), device_id);
        assert_eq!(distributor.prev(), None);

        assert_eq!(*distributor.next(), device_id);
        distributor.select();
        assert_eq!(distributor.prev(), Some(device_id));

        // Always the same
        assert_eq!(*distributor.next(), device_id);
        distributor.select();
        assert_eq!(*distributor.next(), device_id);
    }

    #[test]
    fn test_round_robin_device_selection() {
        let mut distributor = RoundRobinDistributor::new(vec![0, 1, 2]);

        // Nothing selected yet
        assert_eq!(distributor.prev(), None);

        // Predict next, but not marked as selected/used
        assert_eq!(*distributor.next(), 0);
        assert_eq!(distributor.prev(), None);

        // Next is still 0 (stateful, must be marked)
        assert_eq!(*distributor.next(), 0);
        distributor.select();
        assert_eq!(distributor.prev(), Some(0));
        // Next is 1, but prev will still be 0 since it has not been marked
        assert_eq!(*distributor.next(), 1);
        assert_eq!(distributor.prev(), Some(0));
        // Now it's 1
        distributor.select();
        assert_eq!(distributor.prev(), Some(1));
        // Wrapping back to 0
        assert_eq!(*distributor.next(), 2);
        distributor.select();
        assert_eq!(distributor.prev(), Some(2));
        distributor.select();
        assert_eq!(*distributor.next(), 0);
        // Multiple select should not impact the next
        distributor.select();
        distributor.select();
        distributor.select();
        assert_eq!(*distributor.next(), 1);
    }
}
