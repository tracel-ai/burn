use burn_tensor::Device;

/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<I, O>: Send + Sync {
    /// Batches the given items on the specified device.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to batch.
    /// * `device` - The backend device to use.
    ///
    /// # Returns
    ///
    /// The batched items.
    fn batch(&self, items: Vec<I>, device: &Device) -> O;
}

/// Test batcher
#[cfg(test)]
#[derive(new, Clone)]
pub struct TestBatcher;

#[cfg(test)]
impl<I> Batcher<I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>, _device: &Device) -> Vec<I> {
        items
    }
}
