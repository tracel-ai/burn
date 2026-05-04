use burn_tensor::{Device, backend::Backend};

#[cfg(test)]
use crate::TestBackend;

/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<B: Backend, I, O>: Send + Sync {
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
    fn batch(&self, items: Vec<I>, device: &Device<B>) -> O;
}

/// Test batcher
#[cfg(test)]
#[derive(new, Clone)]
pub struct TestBatcher;

#[cfg(test)]
impl<I> Batcher<TestBackend, I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>, _device: &Device<TestBackend>) -> Vec<I> {
        items
    }
}
