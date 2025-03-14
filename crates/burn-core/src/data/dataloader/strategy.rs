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

    /// Creates a new strategy of the same type.
    fn clone_dyn(&self) -> Box<dyn DistributionStrategy<Resource = Self::Resource>>;
}

/// Always selects the same resource or device.
#[derive(Clone)]
pub struct FixedDistributor<R> {
    resources: Vec<R>,
    fixed_id: usize,
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

    /// Create a new fixed distribution strategy. Always selects the first resource.
    /// To change the fixed resource, use `with_fixed(...)`.
    fn new(resources: Vec<Self::Resource>) -> Self
    where
        Self: Sized,
    {
        Self {
            resources,
            fixed_id: 0,
        }
    }

    fn next_id(&mut self) -> usize {
        self.fixed_id
    }

    fn clone_dyn(&self) -> Box<dyn DistributionStrategy<Resource = R>> {
        Box::new(Self::new(self.resources.clone()).with_fixed(self.fixed_id))
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
        let fixed_id = 1;
        let device = 2;
        let devices = vec![0, device];
        let mut distributor = FixedDistributor::new(devices.clone()).with_fixed(fixed_id);

        // Always the same
        for _ in 0..5 {
            assert_eq!(distributor.next_id(), fixed_id);
            assert_eq!(*distributor.next(), device);
        }

        assert_eq!(distributor.resources(), devices.as_slice())
    }
}
